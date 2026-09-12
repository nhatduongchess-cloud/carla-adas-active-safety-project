"""Rút gọn detection đã fuse thành traffic-control context cho planner."""


class StopSignState:
    """Giữ xe dừng đủ thời gian rồi chống kích hoạt lặp khi biển còn trong ảnh."""

    def __init__(self, hold_s=1.5, cooldown_s=8.0, stopped_speed_ms=0.25):
        self.hold_s = float(hold_s)
        self.cooldown_s = float(cooldown_s)
        self.stopped_speed_ms = float(stopped_speed_ms)
        self.pending = False
        self.hold_elapsed = 0.0
        self.cooldown = 0.0

    def update(self, detected, speed_ms, dt):
        dt = max(0.0, float(dt))
        self.cooldown = max(0.0, self.cooldown - dt)
        if detected and self.cooldown <= 0.0:
            self.pending = True
        if self.pending:
            if float(speed_ms) <= self.stopped_speed_ms:
                self.hold_elapsed += dt
            else:
                self.hold_elapsed = 0.0
            if self.hold_elapsed + 1e-9 >= self.hold_s:
                self.pending = False
                self.hold_elapsed = 0.0
                self.cooldown = self.cooldown_s
        return self.pending


def summarize_traffic_controls(detections, light_range_m=45.0, stop_range_m=18.0):
    lights = []
    stop_sign = False
    nearest_stop = None
    for detection in detections:
        cls = detection.get("class")
        distance = detection.get("distance_m")
        if distance is None:
            continue
        distance = float(distance)
        if cls == "TrafficLight" and distance <= light_range_m:
            state = str(detection.get("traffic_light_state", "unknown")).lower()
            lights.append((distance, state))
        elif cls == "StopSign" and distance <= stop_range_m:
            stop_sign = True
            nearest_stop = distance if nearest_stop is None else min(nearest_stop, distance)

    lights.sort(key=lambda item: item[0])
    light_distance, light_state = (lights[0] if lights else (None, "unknown"))
    return {
        "traffic_light_state": light_state,
        "traffic_light_distance_m": light_distance,
        "stop_sign": stop_sign,
        "stop_sign_distance_m": nearest_stop,
    }
