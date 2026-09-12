"""Combined longitudinal and lateral vehicle controller with rate limiting."""

import carla


class VehicleController:
    """Merges longitudinal PID and lateral Stanley control into final CARLA vehicle commands."""

    def __init__(self, max_steering_delta=0.05, max_pedal_delta=0.08) -> None:
        self.previous_steering: float = 0.0
        self.previous_throttle: float = 0.0
        self.previous_brake: float = 0.0
        self.max_steering_delta: float = max_steering_delta
        self.max_pedal_delta: float = max_pedal_delta

    @staticmethod
    def _rate_limit(target, previous, max_delta):
        delta = target - previous
        if abs(delta) <= max_delta:
            return target
        return previous + (max_delta if delta > 0.0 else -max_delta)

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
        raw_steering = self._rate_limit(
            float(raw_steering), self.previous_steering, self.max_steering_delta)
        target_throttle = self._rate_limit(
            float(target_throttle), self.previous_throttle, self.max_pedal_delta)
        target_brake = self._rate_limit(
            float(target_brake), self.previous_brake, self.max_pedal_delta)
        if target_brake > 0.02:
            target_throttle = 0.0
        self.previous_steering = raw_steering
        self.previous_throttle = target_throttle
        self.previous_brake = target_brake

        control_command = carla.VehicleControl()
        control_command.throttle = float(target_throttle)
        control_command.brake = float(target_brake)
        control_command.steer = float(raw_steering)
        control_command.hand_brake = False
        control_command.reverse = False

        return control_command
