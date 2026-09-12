"""PID dọc thuần Python, có anti-windup và vùng chết chống giật ga/phanh."""


class LongitudinalController:
    def __init__(self, kp=0.45, ki=0.04, kd=0.08, integral_limit=8.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, target_speed_ms, current_speed_ms, dt):
        dt = max(1e-3, float(dt))
        error = float(target_speed_ms) - float(current_speed_ms)
        self.integral = max(-self.integral_limit,
                            min(self.integral_limit, self.integral + error * dt))
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        command = self.kp * error + self.ki * self.integral + self.kd * derivative
        if abs(error) < 0.15:
            return 0.0, 0.0
        if command >= 0.0:
            return min(1.0, command), 0.0
        return 0.0, min(1.0, -command)
