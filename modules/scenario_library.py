"""Thư viện kịch bản kiểm thử — PORT 15 kịch bản từ CARLA ScenarioRunner.

Mỗi kịch bản được tái hiện thành "công thức" ĐỘC LẬP, đặt vật cản/actor tương đối
theo LÀN của ego (dùng waypoint API), nên chạy được ở mọi town và KHÔNG cần chạy
ScenarioRunner (tránh lệch phiên bản 0.9.13 vs CARLA của bạn).

Metadata (CATALOG) không import carla ở cấp module -> selftest kiểm được không cần
CARLA. carla chỉ được import BÊN TRONG builder khi thực sự spawn.

Ghi chú: các kịch bản ngã tư/rẽ/ngược chiều là XẤP XỈ di động của bản gốc (đặt xe
cắt vào quỹ đạo ego), giữ đúng giá trị kiểm thử: ego có PHÁT HIỆN + PHANH/NÉ kịp
mà không va chạm hay không.
"""
# fmt: off
# isort: skip_file
import math
from dataclasses import dataclass
from typing import Callable, List


# ============================================================================ #
# Hạ tầng chạy kịch bản
# ============================================================================ #
class RunningScenario:
    def __init__(self, actors, trigger_distance=30.0, on_trigger=None, on_tick=None):
        self.actors = [a for a in actors if a is not None]
        self.trigger_distance = trigger_distance
        self._on_trigger = on_trigger
        self._on_tick = on_tick
        self.triggered = False

    def _min_dist(self, ego):
        el = ego.get_location()
        best = math.inf
        for a in self.actors:
            try:
                if not a.is_alive:
                    continue
                al = a.get_location()
                best = min(best, ((el.x - al.x) ** 2 + (el.y - al.y) ** 2) ** 0.5)
            except Exception:
                pass
        return best

    def tick(self, frame, ego, world):
        try:
            if self._on_tick:
                self._on_tick(self, frame, ego, world)
            if not self.triggered and self._min_dist(ego) < self.trigger_distance:
                self.triggered = True
                if self._on_trigger:
                    self._on_trigger(self, ego, world)
        except Exception:
            pass

    def destroy(self, client):
        import carla
        alive = [a for a in self.actors if a is not None and a.is_alive]
        if alive:
            client.apply_batch([carla.command.DestroyActor(a) for a in alive])
        self.actors = []


@dataclass
class ScenarioSpec:
    name: str
    category: str          # lead | crossing | cutin | junction | oncoming
    tests: str             # chức năng được kiểm
    description: str
    builder: Callable      # (world, ego, tm) -> RunningScenario


# ============================================================================ #
# Helper spawn (import carla cục bộ)
# ============================================================================ #
def _vc(**kw):
    import carla
    return carla.VehicleControl(**kw)


def _spawn_vehicle(world, transform, model='vehicle.tesla.model3'):
    import carla
    bl = world.get_blueprint_library()
    try:
        bp = bl.find(model)
    except Exception:
        bp = bl.filter('vehicle.*')[0]
    if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', 'scenario')
    loc = carla.Location(transform.location.x, transform.location.y, transform.location.z + 0.3)
    return world.try_spawn_actor(bp, carla.Transform(loc, transform.rotation))


def _wp_ahead(world, ego, dist):
    wp = world.get_map().get_waypoint(ego.get_location())
    nxt = wp.next(dist) if wp else None
    return nxt[0] if nxt else None


def _spawn_ahead(world, ego, dist, model='vehicle.tesla.model3'):
    wp = _wp_ahead(world, ego, dist)
    return _spawn_vehicle(world, wp.transform, model) if wp else None


def _spawn_adjacent(world, ego, dist, side, model='vehicle.tesla.model3'):
    """side: 'left' | 'right' — spawn ở làn kế bên, cách ego 'dist' m về phía trước."""
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None
    lane = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
    if lane is None:
        return None
    return _spawn_vehicle(world, lane.transform, model)


def _spawn_crossing(world, ego, dist, side, model='vehicle.tesla.model3'):
    """Xe đặt lệch ngang, quay đầu cắt ngang quỹ đạo ego (xấp xỉ tình huống ngã tư)."""
    import carla
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None
    tf = wp.transform
    right = tf.get_right_vector()
    sgn = 1.0 if side == 'right' else -1.0
    loc = carla.Location(tf.location.x + right.x * 6.0 * sgn,
                         tf.location.y + right.y * 6.0 * sgn,
                         tf.location.z)
    rot = carla.Rotation(yaw=tf.rotation.yaw - 90.0 * sgn)
    return _spawn_vehicle(world, carla.Transform(loc, rot), model)


def _spawn_walker(world, ego, dist, side='right'):
    import carla
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None, None
    tf = wp.transform
    right = tf.get_right_vector()
    sgn = 1.0 if side == 'right' else -1.0
    loc = carla.Location(tf.location.x + right.x * 3.5 * sgn,
                         tf.location.y + right.y * 3.5 * sgn,
                         tf.location.z + 1.0)
    bl = world.get_blueprint_library()
    walkers = bl.filter('walker.pedestrian.*')
    if not walkers:
        return None, None
    walker = world.try_spawn_actor(walkers[0], carla.Transform(loc))
    return walker, (right, sgn)


def _brake_all(sc, level=0.6, hard=False):
    for a in sc.actors:
        try:
            a.set_autopilot(False)
            a.apply_control(_vc(brake=1.0 if hard else level, hand_brake=hard))
        except Exception:
            pass


# ============================================================================ #
# Builders — 15 kịch bản
# ============================================================================ #
def _follow_leading(world, ego, tm):
    lead = _spawn_ahead(world, ego, 18.0)
    if lead:
        lead.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(lead, 30)
    return RunningScenario([lead], trigger_distance=18.0,
                           on_trigger=lambda sc, e, w: _brake_all(sc, level=0.5))


def _follow_leading_obstacle(world, ego, tm):
    lead = _spawn_ahead(world, ego, 16.0)
    obstacle = _spawn_ahead(world, ego, 38.0)
    if lead:
        lead.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(lead, 30)
    if obstacle:
        obstacle.apply_control(_vc(brake=1.0, hand_brake=True))

    def on_trigger(sc, e, w):
        # Chỉ xe dẫn đầu phanh; vật cản đã đứng yên sẵn.
        if lead is not None:
            try:
                lead.set_autopilot(False)
                lead.apply_control(_vc(brake=0.6))
            except Exception:
                pass
    return RunningScenario([lead, obstacle], trigger_distance=18.0, on_trigger=on_trigger)


def _other_leading(world, ego, tm):
    lead = _spawn_ahead(world, ego, 16.0)
    side = _spawn_adjacent(world, ego, 20.0, 'left')
    for a in (lead, side):
        if a:
            a.set_autopilot(True, tm.get_port())
    return RunningScenario([lead, side], trigger_distance=16.0,
                           on_trigger=lambda sc, e, w: _brake_all(sc, hard=True))


def _hard_brake(world, ego, tm):
    lead = _spawn_ahead(world, ego, 15.0)
    if lead:
        lead.set_autopilot(True, tm.get_port())
    return RunningScenario([lead], trigger_distance=14.0,
                           on_trigger=lambda sc, e, w: _brake_all(sc, hard=True))


def _stationary_object(world, ego, tm):
    obj = _spawn_ahead(world, ego, 22.0, model='vehicle.diamondback.century')  # xe đạp đứng yên
    if obj is None:
        obj = _spawn_ahead(world, ego, 22.0)
    if obj:
        obj.apply_control(_vc(brake=1.0, hand_brake=True))
    return RunningScenario([obj], trigger_distance=99.0)  # tĩnh, không cần trigger


def _dynamic_object(world, ego, tm):
    walker, meta = _spawn_walker(world, ego, 20.0, side='right')

    def cross(sc, e, w):
        import carla
        if walker is None or meta is None:
            return
        right, sgn = meta
        ctrl = carla.WalkerControl()
        ctrl.direction = carla.Vector3D(x=-right.x * sgn, y=-right.y * sgn, z=0.0)
        ctrl.speed = 1.6
        try:
            walker.apply_control(ctrl)
        except Exception:
            pass
    return RunningScenario([walker], trigger_distance=15.0, on_trigger=cross)


def _construction(world, ego, tm):
    import carla
    actors = []
    blocker = _spawn_ahead(world, ego, 26.0)
    if blocker:
        blocker.apply_control(_vc(brake=1.0, hand_brake=True))
        actors.append(blocker)
    bl = world.get_blueprint_library()
    cones = bl.filter('static.prop.trafficcone01') or bl.filter('static.prop.constructioncone')
    if cones:
        for d in (16.0, 20.0, 23.0):
            wp = _wp_ahead(world, ego, d)
            if wp:
                actors.append(world.try_spawn_actor(cones[0], wp.transform))
    return RunningScenario(actors, trigger_distance=99.0)


def _cut_in(side):
    def build(world, ego, tm):
        actor = _spawn_adjacent(world, ego, 12.0, side)
        if actor:
            actor.set_autopilot(True, tm.get_port())

        def do_cut(sc, e, w):
            for a in sc.actors:
                try:
                    # side='left' -> cắt sang phải (True); side='right' -> cắt sang trái (False)
                    tm.force_lane_change(a, side == 'left')
                except Exception:
                    pass
        return RunningScenario([actor], trigger_distance=18.0, on_trigger=do_cut)
    return build


def _highway_cut_in(world, ego, tm):
    actor = _spawn_adjacent(world, ego, 14.0, 'left')
    if actor:
        actor.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(actor, -30)  # nhanh hơn (cao tốc)

    def do_cut(sc, e, w):
        for a in sc.actors:
            try:
                tm.force_lane_change(a, True)
            except Exception:
                pass
    return RunningScenario([actor], trigger_distance=22.0, on_trigger=do_cut)


def _vehicle_turning(side):
    def build(world, ego, tm):
        actor = _spawn_crossing(world, ego, 24.0, side)
        return RunningScenario([actor], trigger_distance=20.0,
                               on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.5)) for a in sc.actors if a])
    return build


def _no_signal_junction(world, ego, tm):
    actor = _spawn_crossing(world, ego, 26.0, 'left')
    return RunningScenario([actor], trigger_distance=22.0,
                           on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.6)) for a in sc.actors if a])


def _oncoming(world, ego, tm):
    import carla
    wp = _wp_ahead(world, ego, 45.0)
    actor = None
    if wp:
        tf = wp.transform
        rot = carla.Rotation(yaw=tf.rotation.yaw + 180.0)
        actor = _spawn_vehicle(world, carla.Transform(tf.location, rot))
    return RunningScenario([actor], trigger_distance=40.0,
                           on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.5)) for a in sc.actors if a])


def _red_light_crosser(world, ego, tm):
    actor = _spawn_crossing(world, ego, 28.0, 'left')
    return RunningScenario([actor], trigger_distance=26.0,
                           on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.9)) for a in sc.actors if a])


# ============================================================================ #
# CATALOG (15) — an toàn để import không cần carla
# ============================================================================ #
CATALOG: List[ScenarioSpec] = [
    ScenarioSpec("FollowLeadingVehicle", "lead", "FOLLOW + AEB",
                 "Xe dẫn đầu giảm tốc; ego phải giữ khoảng cách / giảm tốc.", _follow_leading),
    ScenarioSpec("FollowLeadingVehicleWithObstacle", "lead", "AEB (phanh dừng)",
                 "Xe dẫn đầu + vật cản đứng yên xa hơn; ego phải phanh dừng.", _follow_leading_obstacle),
    ScenarioSpec("OtherLeadingVehicle", "lead", "AEB",
                 "Hai xe dẫn đầu, một xe phanh gấp.", _other_leading),
    ScenarioSpec("HardBrake", "lead", "Phanh khẩn cấp",
                 "Xe dẫn đầu phanh GẤP -> kiểm tra AEB khẩn cấp.", _hard_brake),
    ScenarioSpec("StationaryObjectCrossing", "crossing", "AEB (vật tĩnh)",
                 "Vật đứng yên (xe đạp) chắn làn -> phanh (hành lang LiDAR).", _stationary_object),
    ScenarioSpec("DynamicObjectCrossing", "crossing", "Phanh khẩn cấp (người đi bộ)",
                 "Người đi bộ bất ngờ băng qua -> phanh khẩn cấp.", _dynamic_object),
    ScenarioSpec("ConstructionObstacle", "crossing", "AEB (công trường)",
                 "Chướng ngại công trường (cọc + xe) chắn làn.", _construction),
    ScenarioSpec("CutInFrom_left_Lane", "cutin", "Phản ứng cắt làn / né",
                 "Xe từ làn TRÁI tạt đầu vào làn ego.", _cut_in('left')),
    ScenarioSpec("CutInFrom_right_Lane", "cutin", "Phản ứng cắt làn / né",
                 "Xe từ làn PHẢI tạt đầu vào làn ego.", _cut_in('right')),
    ScenarioSpec("HighwayCutIn", "cutin", "Cắt làn tốc độ cao",
                 "Xe tạt đầu ở tốc độ cao.", _highway_cut_in),
    ScenarioSpec("VehicleTurningRight", "junction", "Phát hiện + phanh ở giao lộ",
                 "Xe cắt ngang quỹ đạo ego từ bên phải (xấp xỉ ngã tư).", _vehicle_turning('right')),
    ScenarioSpec("VehicleTurningLeft", "junction", "Phát hiện + phanh ở giao lộ",
                 "Xe cắt ngang quỹ đạo ego từ bên trái (xấp xỉ ngã tư).", _vehicle_turning('left')),
    ScenarioSpec("NoSignalJunctionCrossing", "junction", "Phanh cho xe cắt ngang",
                 "Xe cắt ngang ở ngã tư không đèn.", _no_signal_junction),
    ScenarioSpec("ManeuverOppositeDirection", "oncoming", "Phát hiện + phanh xe ngược chiều",
                 "Xe ngược chiều tiến tới (xấp xỉ tình huống vượt/đối đầu).", _oncoming),
    ScenarioSpec("OppositeVehicleRunningRedLight", "oncoming", "Phanh khẩn cấp giao lộ",
                 "Xe vượt đèn đỏ lao cắt ngang ở tốc độ cao.", _red_light_crosser),
]


def list_names():
    return [s.name for s in CATALOG]


def get(name):
    for s in CATALOG:
        if s.name == name:
            return s
    return None
