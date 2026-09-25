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

try:
    from clearance import clearance as _surface_clearance, weakest_basis
except ImportError:
    from modules.clearance import clearance as _surface_clearance, weakest_basis


# ============================================================================ #
# Hạ tầng chạy kịch bản
# ============================================================================ #
class RunningScenario:
    def __init__(self, actors, trigger_distance=30.0, on_trigger=None, on_tick=None,
                 reaction_condition=None):
        self.actors = [a for a in actors if a is not None]
        self.trigger_distance = trigger_distance
        self._on_trigger = on_trigger
        self._on_tick = on_tick
        self._reaction_condition = reaction_condition
        self.triggered = False
        self.trigger_frame = None
        # ``trigger_frame`` is when the scenario starts commanding its actor.
        # For a cut-in, the actor can still be fully inside the adjacent lane
        # for several seconds. ``hazard_frame`` is the ground-truth instant at
        # which the hazard becomes relevant to the ego swept corridor and is
        # therefore the correct origin for reaction-latency acceptance.
        self.hazard_frame = None
        # Centre-to-centre distance: what triggers the scenario. Kept exactly as
        # before, so no scenario starts earlier or later than it used to.
        self.min_actor_distance_m = math.inf
        # Surface-to-surface clearance: what the acceptance criterion reads.
        # See modules/clearance.py for why these are two different numbers.
        self.min_clearance_m = math.inf
        self.clearance_basis = None
        self._bases_seen = []

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

    def _min_clearance(self, ego):
        """Smallest surface-to-surface gap to any live actor, and its basis."""
        best, basis = math.inf, None
        for a in self.actors:
            try:
                if not a.is_alive:
                    continue
                gap, how = _surface_clearance(ego, a)
                if gap < best:
                    best, basis = gap, how
            except Exception:
                pass
        return best, basis

    def tick(self, frame, ego, world):
        try:
            if self._on_tick:
                self._on_tick(self, frame, ego, world)
            distance = self._min_dist(ego)
            self.min_actor_distance_m = min(self.min_actor_distance_m, distance)
            gap, basis = self._min_clearance(ego)
            if basis is not None:
                self._bases_seen.append(basis)
                self.clearance_basis = weakest_basis(self._bases_seen)
                self._bases_seen = [self.clearance_basis]
            self.min_clearance_m = min(self.min_clearance_m, gap)
            if not self.triggered and distance < self.trigger_distance:
                self.triggered = True
                self.trigger_frame = int(frame)
                if self._on_trigger:
                    self._on_trigger(self, ego, world)
            if self.triggered and self.hazard_frame is None:
                hazard_active = (True if self._reaction_condition is None else
                                 bool(self._reaction_condition(self, ego, world)))
                if hazard_active:
                    self.hazard_frame = int(frame)
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


def _spawn_adjacent(world, ego, dist, side, model='vehicle.tesla.model3',
                    allow_geometric_fallback=True):
    """side: 'left' | 'right' — spawn ở làn kế bên, cách ego 'dist' m về phía trước."""
    import carla
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None
    lane = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
    # ``get_left/right_lane`` can return an opposing or shoulder lane. Handing
    # that actor to Traffic Manager's force_lane_change makes it pass the ego
    # in the opposite direction instead of performing a cut-in. Only accept a
    # same-direction driving lane; this rule is topology-based and town-agnostic.
    valid_lane = (lane is not None
                  and lane.lane_type == carla.LaneType.Driving
                  and lane.lane_id * wp.lane_id > 0)
    actor = _spawn_vehicle(world, lane.transform, model) if valid_lane else None
    if actor is not None:
        return actor
    if not allow_geometric_fallback:
        return None

    # Một số đoạn Town02 chỉ có lane topology ở một phía; recipe right-cut-in
    # trước đây trả actor=None và test "PASS/FAIL" chỉ đo một scenario rỗng.
    # Fallback hình học vẫn đặt actor song song với ego, rồi Traffic Manager ép
    # nó cắt vào làn. Thử từ ngoài vào trong để tránh vật thể ven đường.
    tf = wp.transform
    right = tf.get_right_vector()
    sign = -1.0 if side == 'left' else 1.0
    for lateral_m in (4.2, 3.6, 3.0):
        loc = carla.Location(tf.location.x + right.x * lateral_m * sign,
                             tf.location.y + right.y * lateral_m * sign,
                             tf.location.z)
        actor = _spawn_vehicle(world, carla.Transform(loc, tf.rotation), model)
        if actor is not None:
            return actor
    return None


def _spawn_staged_lateral(world, ego, dist, side,
                          model='vehicle.tesla.model3'):
    """Stage-spawn then teleport an actor when a side has no driving lane.

    CARLA correctly rejects a normal vehicle spawn on Town02's 0.3 m shoulder.
    A staged actor lets the portable right-cut-in recipe remain observable; it
    is driven manually by the builder and never handed to Traffic Manager.
    """
    import carla
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None
    tf = wp.transform
    right = tf.get_right_vector()
    sign = -1.0 if side == 'left' else 1.0
    # Stage approximately one lane-width away. The old 0.68× offset placed a
    # normal car's bounding box inside the ego corridor at frame zero, creating
    # an physically unavoidable 12 m side impact instead of a genuine cut-in.
    lateral_m = max(3.2, wp.lane_width * 0.95)
    target = carla.Transform(
        carla.Location(tf.location.x + right.x * lateral_m * sign,
                       tf.location.y + right.y * lateral_m * sign,
                       tf.location.z + 0.6),
        tf.rotation)

    bl = world.get_blueprint_library()
    try:
        bp = bl.find(model)
    except Exception:
        bp = bl.filter('vehicle.*')[0]
    if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', 'scenario')
    target_loc = target.location
    for staging in reversed(world.get_map().get_spawn_points()):
        if staging.location.distance(target_loc) < 30.0:
            continue
        actor = world.try_spawn_actor(bp, staging)
        if actor is None:
            continue
        actor.set_simulate_physics(False)
        actor.set_transform(target)
        actor.set_simulate_physics(True)
        return actor
    return None


def _spawn_crossing(world, ego, dist, side, model='vehicle.tesla.model3'):
    """Xe đặt lệch ngang, quay đầu cắt ngang quỹ đạo ego (xấp xỉ tình huống ngã tư)."""
    import carla
    wp = _wp_ahead(world, ego, dist)
    if wp is None:
        return None
    tf = wp.transform
    right = tf.get_right_vector()
    sgn = 1.0 if side == 'right' else -1.0
    rot = carla.Rotation(yaw=tf.rotation.yaw - 90.0 * sgn)
    # 6 m có thể nằm trong nhà/vỉa hè ở town nhỏ và khiến try_spawn_actor trả
    # None. Giảm dần offset nhưng giữ hướng cắt ngang để recipe luôn có actor.
    for lateral_m in (5.0, 4.0, 3.0, 2.0, 0.0):
        loc = carla.Location(tf.location.x + right.x * lateral_m * sgn,
                             tf.location.y + right.y * lateral_m * sgn,
                             tf.location.z)
        actor = _spawn_vehicle(world, carla.Transform(loc, rot), model)
        if actor is not None:
            return actor
    return None


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


def _overlaps_ego_corridor(sc, ego, _world, corridor_half_m=1.75,
                           lookahead_m=40.0):
    """Ground-truth cut-in activation, independent of CARLA town topology.

    The actor oriented bounding box is projected into the current ego frame.
    A cut-in becomes a safety hazard when any part of that box overlaps the
    ego swept corridor ahead; merely requesting a lane change is not yet a
    detectable/in-path hazard and must not start the reaction timer.
    """
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = math.radians(ego_tf.rotation.yaw)
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
    for actor in sc.actors:
        try:
            if not actor.is_alive:
                continue
            actor_tf = actor.get_transform()
            dx = actor_tf.location.x - ego_loc.x
            dy = actor_tf.location.y - ego_loc.y
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            delta_yaw = math.radians(actor_tf.rotation.yaw - ego_tf.rotation.yaw)
            extent = actor.bounding_box.extent
            lateral_reach = (abs(math.sin(delta_yaw)) * float(extent.x)
                             + abs(math.cos(delta_yaw)) * float(extent.y))
            longitudinal_reach = (abs(math.cos(delta_yaw)) * float(extent.x)
                                  + abs(math.sin(delta_yaw)) * float(extent.y))
            overlaps_ahead = (local_x + longitudinal_reach > 0.0
                              and local_x - longitudinal_reach < lookahead_m)
            overlaps_lateral = abs(local_y) - lateral_reach <= corridor_half_m
            if overlaps_ahead and overlaps_lateral:
                return True
        except Exception:
            continue
    return False


# ============================================================================ #
# Builders — 15 kịch bản
# ============================================================================ #
def _follow_leading(world, ego, tm):
    lead = _spawn_ahead(world, ego, 18.0)
    if lead:
        lead.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(lead, 60)
    return RunningScenario([lead], trigger_distance=18.0,
                           on_trigger=lambda sc, e, w: _brake_all(sc, level=0.5))


def _follow_leading_obstacle(world, ego, tm):
    lead = _spawn_ahead(world, ego, 16.0)
    obstacle = _spawn_ahead(world, ego, 38.0)
    if lead:
        lead.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(lead, 60)
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
    # Khoảng cách Euclid lúc spawn ≈20.3m. Trigger ngay để walker bắt đầu băng
    # qua; nếu chờ 15m, VRU safety có thể dừng ego trước trigger và gây deadlock.
    return RunningScenario([walker], trigger_distance=22.0, on_trigger=cross)


def _construction(world, ego, tm):
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


def _cut_in(side, cut_in_distance_m=18.0, staged_speed_ms=6.0,
            transition_s=1.5, tm_speed_diff=None):
    def build(world, ego, tm):
        # A cut-in must start from a real same-direction adjacent driving lane.
        # When a town has only an opposing lane/shoulder on that side, switch
        # to the deterministic staged/manual recipe instead of asking Traffic
        # Manager to perform an impossible lane change.
        # Default 18 m is inside the requested 10--20 m AEB validation band
        # while leaving a physically meaningful stopping/evasion envelope.
        actor = _spawn_adjacent(
            world, ego, cut_in_distance_m, side,
            allow_geometric_fallback=False)
        manual_cut = actor is None
        if manual_cut:
            actor = _spawn_staged_lateral(world, ego, cut_in_distance_m, side)
            if actor is not None:
                # Portable fallback uses a deterministic kinematic trajectory.
                # Physics on a non-driving shoulder is town-dependent (curbs,
                # props and opposing lanes can make manual steering stick or
                # ram the ego later), so do not let map collision geometry alter
                # the intended test path.
                actor.set_simulate_physics(False)
        if actor and not manual_cut:
            actor.set_autopilot(True, tm.get_port())
            if tm_speed_diff is not None:
                tm.vehicle_percentage_speed_difference(actor, tm_speed_diff)

        cut_base_wp = _wp_ahead(world, ego, cut_in_distance_m)
        staged_lateral_m = (max(3.2, cut_base_wp.lane_width * 0.95)
                            if cut_base_wp is not None else 3.5)

        def do_cut(sc, e, w):
            for a in sc.actors:
                try:
                    # side='left' -> cắt sang phải (True); side='right' -> cắt sang trái (False)
                    if not manual_cut:
                        tm.force_lane_change(a, side == 'left')
                except Exception:
                    pass

        def continue_cut(sc, frame, e, w):
            if not manual_cut or not sc.triggered:
                return
            elapsed = frame - sc.trigger_frame
            t_s = max(0.0, elapsed * 0.025)
            u = min(1.0, t_s / transition_s)
            smooth = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
            lateral = staged_lateral_m * (1.0 - smooth)
            lateral *= -1.0 if side == 'left' else 1.0
            forward_m = staged_speed_ms * t_s
            center_wp = cut_base_wp
            if cut_base_wp is not None and forward_m > 1e-6:
                candidates = cut_base_wp.next(forward_m)
                if candidates:
                    center_wp = min(
                        candidates,
                        key=lambda wp: abs((wp.transform.rotation.yaw
                                            - cut_base_wp.transform.rotation.yaw
                                            + 180.0) % 360.0 - 180.0))
            if center_wp is None:
                return
            import carla
            center_tf = center_wp.transform
            right = center_tf.get_right_vector()
            location = carla.Location(
                center_tf.location.x + right.x * lateral,
                center_tf.location.y + right.y * lateral,
                center_tf.location.z + 0.3)
            transform = carla.Transform(location, center_tf.rotation)
            for a in sc.actors:
                a.set_transform(transform)

        return RunningScenario(
            [actor], trigger_distance=cut_in_distance_m + 6.0, on_trigger=do_cut,
            on_tick=continue_cut, reaction_condition=_overlaps_ego_corridor)
    return build


def _highway_cut_in(world, ego, tm):
    # Reuse the validated topology-aware recipe. On a real same-direction
    # adjacent lane Traffic Manager drives 30% faster; otherwise the portable
    # kinematic fallback moves at 10 m/s and completes the cut in 1.2 s.
    return _cut_in(
        'left', cut_in_distance_m=20.0, staged_speed_ms=10.0,
        transition_s=1.2, tm_speed_diff=-30)(world, ego, tm)


def _vehicle_turning(side):
    def build(world, ego, tm):
        actor = _spawn_crossing(world, ego, 24.0, side)
        # Trigger ngay sau spawn (initial Euclidean distance khoảng 24-25 m),
        # trước khi ego dừng vì nhìn thấy actor tĩnh.
        return RunningScenario([actor], trigger_distance=32.0,
                               on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.5)) for a in sc.actors if a])
    return build


def _no_signal_junction(world, ego, tm):
    actor = _spawn_crossing(world, ego, 26.0, 'left')
    return RunningScenario([actor], trigger_distance=34.0,
                           on_trigger=lambda sc, e, w: [a.apply_control(_vc(throttle=0.6)) for a in sc.actors if a])


def _oncoming(world, ego, tm):
    import carla
    wp = _wp_ahead(world, ego, 45.0)
    actor = None
    if wp:
        tf = wp.transform
        rot = carla.Rotation(yaw=tf.rotation.yaw + 180.0)
        actor = _spawn_vehicle(world, carla.Transform(tf.location, rot))
    def approach(sc, e, w):
        for a in sc.actors:
            if a:
                a.apply_control(_vc(throttle=0.5))

    def yield_before_contact(sc, frame, e, w):
        # Ego vẫn phải nhận diện/phanh từ 40 m. Actor đối đầu cũng phanh khi còn
        # 14 m để scenario kiểm AEB thay vì cố tình đâm một ego đã safe-stop.
        if sc.triggered and sc._min_dist(e) < 14.0:
            _brake_all(sc, hard=True)

    return RunningScenario([actor], trigger_distance=40.0,
                           on_trigger=approach, on_tick=yield_before_contact)


def _red_light_crosser(world, ego, tm):
    actor = _spawn_crossing(world, ego, 28.0, 'left')
    return RunningScenario([actor], trigger_distance=36.0,
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
