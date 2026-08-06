"""Bộ điều khiển ego dùng chung: dịch quyết định an toàn + L3 thành lệnh CARLA.

Tách ra để chinh.py / evaluate_l3.py / run_scenarios.py không lặp lại logic. Ưu
tiên: L3 override (MRM) > AEB (BRAKE) > lái thường (autopilot + TM). Chỉ bật/tắt
autopilot khi ĐỔI chế độ (nhờ arbiter đã có cam kết -> không rung).
"""
# fmt: off
# isort: skip_file
import carla


def spawn_ego_safe(world, blueprint, spawn_points=None, preferred_index=0):
    """Spawn ego ở điểm TRỐNG đầu tiên (thử preferred trước rồi tới các điểm khác).

    Tránh lỗi 'Spawn failed because of collision at spawn position' khi điểm spawn
    bị actor khác chiếm. Trả về (ego, index).
    """
    if spawn_points is None:
        spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Bản đồ không có spawn point nào.")
    order = list(range(len(spawn_points)))
    if 0 <= preferred_index < len(spawn_points):
        order.remove(preferred_index)
        order.insert(0, preferred_index)
    for i in order:
        ego = world.try_spawn_actor(blueprint, spawn_points[i])
        if ego is not None:
            return ego, i
    raise RuntimeError("Mọi spawn point đều bị chiếm — chạy lại với '--clean' để dọn actor cũ.")


def destroy_all_actors(world, client):
    """Dọn mọi xe / người đi bộ / cảm biến còn sót (từ các lần chạy trước)."""
    victims = []
    for pat in ('vehicle.*', 'walker.*', 'sensor.*', 'controller.ai.walker'):
        victims += list(world.get_actors().filter(pat))
    for a in victims:
        try:
            if a.type_id.startswith('sensor.') and a.is_alive:
                a.stop()
        except Exception:
            pass
    if victims:
        client.apply_batch([carla.command.DestroyActor(a) for a in victims])
    return len(victims)


def set_hazard_lights(vehicle, on: bool) -> None:
    try:
        s = carla.VehicleLightState(
            carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker) \
            if on else carla.VehicleLightState.NONE
        vehicle.set_light_state(s)
    except Exception:
        pass


class EgoController:
    def __init__(self, ego, traffic_manager, cfg):
        self.ego = ego
        self.tm = traffic_manager
        self.cfg = cfg
        self.engaged = True
        self.last_lane_change_frame = -(10 ** 9)
        self.cooldown_frames = cfg.LANE_CHANGE_COOLDOWN_S / cfg.FIXED_DELTA

    def apply(self, decision, l3, odd_state, frame):
        ego, tm, cfg = self.ego, self.tm, self.cfg

        if l3['override']:
            if self.engaged:
                ego.set_autopilot(False)
                self.engaged = False
            bcmd = 1.0 if l3['state'] == 'SAFE_STOP' else min(1.0, l3['target_decel_ms2'] / 6.0)
            ego.apply_control(carla.VehicleControl(
                brake=bcmd, hand_brake=(l3['state'] == 'SAFE_STOP')))
            set_hazard_lights(ego, True)

        elif decision.action == "BRAKE":
            set_hazard_lights(ego, l3['hazard'])
            if self.engaged:
                ego.set_autopilot(False)
                self.engaged = False
            ego.apply_control(carla.VehicleControl(brake=decision.brake))

        else:
            set_hazard_lights(ego, l3['hazard'])
            if not self.engaged:
                ego.set_autopilot(True, tm.get_port())
                self.engaged = True
            drive_diff = 40.0 if odd_state == "DEGRADED" else cfg.TM_DEFAULT_SPEED_DIFF
            if decision.action == "SLOW":
                tm.vehicle_percentage_speed_difference(ego, max(decision.slow_pct, drive_diff))
            elif decision.action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
                tm.vehicle_percentage_speed_difference(ego, drive_diff)
                if frame - self.last_lane_change_frame > self.cooldown_frames:
                    tm.force_lane_change(ego, decision.action == "LANE_CHANGE_RIGHT")
                    self.last_lane_change_frame = frame
            else:  # DRIVE
                tm.vehicle_percentage_speed_difference(ego, drive_diff)
