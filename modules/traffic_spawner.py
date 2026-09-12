"""Sinh và điều phối giao thông NPC để kiểm thử AEB / tránh né.

TrafficSpawner tạo môi trường "sống": nhiều xe NPC chạy tự động qua Traffic
Manager, mỗi xe có hành vi NGẪU NHIÊN (đổi làn trái/phải, nhanh/chậm khác nhau,
đôi khi vượt đèn) để tạo tình huống bất ngờ cho xe ego xử lý. Tùy chọn còn có
thể đặt một xe ĐỨNG YÊN ngay trước mũi ego để ép kích hoạt phanh gấp / chuyển làn.
"""
# fmt: off
# isort: skip_file
import random
import carla


class TrafficSpawner:
    def __init__(self, world, traffic_manager, seed=None, on_spawn=None,
                 strict=False):
        self.world = world
        self.tm = traffic_manager
        self.bpl = world.get_blueprint_library()
        self.rng = random.Random(seed)
        self.on_spawn = on_spawn
        self.strict = bool(strict)
        if seed is not None:
            try:
                self.tm.set_random_device_seed(seed)
            except Exception:
                if self.strict:
                    raise
        self.actors = []

    def _own(self, actor):
        """Register a spawned actor before any later configuration can fail."""
        self.actors.append(actor)
        if self.on_spawn is not None:
            self.on_spawn(actor)

    def _vehicle_blueprints(self):
        """Chỉ lấy xe 4 bánh để giao thông ổn định, tránh xe đạp/2 bánh lỗi vật lý."""
        bps = self.bpl.filter('vehicle.*')
        four_wheels = []
        for bp in bps:
            if bp.has_attribute('number_of_wheels') and \
                    int(bp.get_attribute('number_of_wheels')) == 4:
                four_wheels.append(bp)
        return four_wheels or list(bps)

    def spawn_traffic(self, num_vehicles):
        """Sinh tối đa num_vehicles xe NPC tại các điểm spawn ngẫu nhiên."""
        spawn_points = self.world.get_map().get_spawn_points()
        self.rng.shuffle(spawn_points)
        # Chừa điểm đầu cho ego (chinh.py spawn ego tại spawn_points[0] gốc).
        blueprints = self._vehicle_blueprints()

        count = 0
        for sp in spawn_points:
            if count >= num_vehicles:
                break
            bp = self.rng.choice(blueprints)
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                if colors:
                    bp.set_attribute('color', self.rng.choice(colors))
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', 'autopilot')

            npc = self.world.try_spawn_actor(bp, sp)
            if npc is None:
                continue  # điểm bị chiếm -> bỏ qua
            self._own(npc)
            npc.set_autopilot(True, self.tm.get_port())
            self._randomize(npc)
            count += 1

        print(f"[Traffic] Đã sinh {count}/{num_vehicles} xe NPC (hành vi ngẫu nhiên).")
        return count

    def _randomize(self, v):
        """Gán hành vi ngẫu nhiên cho từng xe qua Traffic Manager."""
        tm = self.tm
        try:
            tm.auto_lane_change(v, True)
            tm.random_left_lanechange_percentage(v, self.rng.randint(10, 60))
            tm.random_right_lanechange_percentage(v, self.rng.randint(10, 60))
            # Âm = chạy nhanh hơn giới hạn, dương = chậm hơn -> đa dạng tốc độ.
            tm.vehicle_percentage_speed_difference(v, self.rng.randint(-40, 40))
            tm.distance_to_leading_vehicle(v, self.rng.uniform(1.0, 3.0))
            tm.ignore_lights_percentage(v, self.rng.randint(0, 30))
            tm.ignore_signs_percentage(v, self.rng.randint(0, 20))
        except Exception as e:
            print(f"[Traffic] Cảnh báo: không áp được hành vi NGẪU nhiên: {e}")
            if self.strict:
                raise

    def spawn_hazard_ahead(self, ego_vehicle, distance_m=20.0):
        """Đặt một xe ĐỨNG YÊN cùng làn, cách ego ~distance_m về phía trước."""
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(ego_vehicle.get_location())
        ahead = ego_wp.next(distance_m)
        if not ahead:
            print("[Traffic] Không tìm được waypoint phía trước để đặt chướng ngại.")
            return None

        transform = ahead[0].transform
        transform.location.z += 0.3
        bp = self.bpl.find('vehicle.audi.etron')
        hazard = self.world.try_spawn_actor(bp, transform)
        if hazard is None:
            print("[Traffic] Vị trí phía trước bị chiếm, không sinh được chướng ngại.")
            return None

        self._own(hazard)
        hazard.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        print(f"[Traffic] ⚠ Đã đặt xe CHƯỚNG NGẠI đứng yên cách ego ~{distance_m:.0f}m.")
        return hazard

    def destroy(self, client):
        """Tiêu hủy toàn bộ NPC + chướng ngại đã sinh."""
        actors = list(self.actors)
        self.actors = []
        if not actors:
            return
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])
            print(f"[Traffic] Đã dọn {len(actors)} actor giao thông.")
        except Exception as exc:
            # The server can be gone after a UE4 device-lost/crash.  Do not
            # raise from finally; the primary runtime exception is more useful.
            print(f"[Traffic] Cảnh báo không thể dọn actor sau mất kết nối: {exc}")
