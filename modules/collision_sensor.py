"""Cảm biến va chạm (CARLA sensor.other.collision) + bộ đếm sự kiện.

Cung cấp chỉ số KHÁCH QUAN để chấm điểm hệ thống: số lần ego thực sự va chạm khi
chạy kịch bản. Một lần chạm vật lý phát ra rất nhiều callback liên tiếp, nên ta
KHỬ DỘI (debounce) theo số frame để mỗi cú va chỉ tính là 1.
"""
# fmt: off
# isort: skip_file
import carla


class CollisionSensor:
    def __init__(self, world, ego_vehicle, fps=20, debounce_s=1.0):
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego_vehicle)

        self.count = 0
        self.last_other = None
        self.last_intensity = 0.0
        self.history = []                       # [{'frame', 'other', 'intensity'}]
        self._debounce_frames = max(1, int(debounce_s * fps))
        self._last_frame = -(10 ** 9)

        # Callback chạy trong thread cảm biến; các thao tác đơn giản an toàn nhờ GIL.
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event):
        imp = event.normal_impulse
        intensity = (imp.x ** 2 + imp.y ** 2 + imp.z ** 2) ** 0.5

        # Bỏ qua các callback dội lại của cùng một cú va (trong cửa sổ debounce).
        if event.frame - self._last_frame <= self._debounce_frames:
            return
        self._last_frame = event.frame

        self.count += 1
        other = event.other_actor
        self.last_other = other.type_id if other is not None else 'unknown'
        self.last_intensity = intensity
        self.history.append({
            'frame': event.frame,
            'other': self.last_other,
            'intensity': intensity,
        })
        print(f"[Collision] ❌ #{self.count} với '{self.last_other}' (impulse {intensity:.0f})")

    def stop(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
