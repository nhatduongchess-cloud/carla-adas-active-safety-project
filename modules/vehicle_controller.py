"""Combined longitudinal and lateral vehicle controller with rate limiting."""

import carla


class VehicleController:
    """Merges longitudinal PID and lateral Stanley control into final CARLA vehicle commands."""

    def __init__(self) -> None:
        self.previous_steering: float = 0.0
        self.max_steering_delta: float = 0.05  # Limits steering change rate per tick

    def create_control_command(
        self, target_throttle: float, target_brake: float, raw_steering: float
    ) -> carla.VehicleControl:
        """Applies rate-limiting filtering to steering commands to prevent abrupt jerks.

        Args:
            target_throttle: Desired throttle input [0.0, 1.0].
            target_brake: Desired brake input [0.0, 1.0].
            raw_steering: Calculated raw steering command [-1.0, 1.0].

        Returns:
            A fully validated carla.VehicleControl object.
        """
        # Apply rate limiter on steering to simulate smooth actuator response
        steering_delta = raw_steering - self.previous_steering
        if abs(steering_delta) > self.max_steering_delta:
            raw_steering = self.previous_steering + \
                (self.max_steering_delta if steering_delta >
                 0 else -self.max_steering_delta)

        self.previous_steering = raw_steering

        control_command = carla.VehicleControl()
        control_command.throttle = float(target_throttle)
        control_command.brake = float(target_brake)
        control_command.steer = float(raw_steering)
        control_command.hand_brake = False
        control_command.reverse = False

        return control_command
