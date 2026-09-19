"""Custom ego route/local-planning and closed-loop vehicle control for CARLA."""

import math

if __package__:
    from .lateral_controller import LateralController
    from .local_planner import LocalPlanner
    from .longitudinal_controller import LongitudinalController
    from .planner import BehaviorPlanner, DrivingState
    from .road_geometry import EgoRoute, get_lane_context
    from .scene_semantics import StopSignState
    from .vehicle_controller import VehicleController
else:
    from lateral_controller import LateralController
    from local_planner import LocalPlanner
    from longitudinal_controller import LongitudinalController
    from planner import BehaviorPlanner, DrivingState
    from road_geometry import EgoRoute, get_lane_context
    from scene_semantics import StopSignState
    from vehicle_controller import VehicleController


class EgoDrivingStack:
    def __init__(self, world, ego, cfg, route=None):
        self.world, self.ego, self.cfg = world, ego, cfg
        self.route = route if route is not None else EgoRoute(world, ego)
        self._route_prepared = False
        self.behavior = BehaviorPlanner()
        self.local = LocalPlanner()
        self.lateral = LateralController()
        self.longitudinal = LongitudinalController(
            cfg.CONTROL_KP, cfg.CONTROL_KI, cfg.CONTROL_KD)
        self.vehicle_control = VehicleController()
        self._lane_change_side = None
        self._lane_change_origin = None
        self.stop_sign = StopSignState(
            cfg.STOP_SIGN_HOLD_S, cfg.STOP_SIGN_COOLDOWN_S)
        self._turn_intent = None

    @staticmethod
    def _speed_ms(vehicle):
        v = vehicle.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def request_lane_change(self, side):
        if side not in ("left", "right"):
            return False
        context = (self.route.context if self._route_prepared
                   else get_lane_context(self.world, self.ego))
        if not context.get(f"{side}_exists") or not context.get(f"{side}_change_allowed"):
            return False
        if self._lane_change_side is None:
            self._lane_change_side = side
            self._lane_change_origin = context.get("current_lane_id")
        return True

    def set_turn_intent(self, turn_intent):
        intent = str(turn_intent or "").lower()
        self._turn_intent = intent if intent in ("left", "right") else None

    def prepare_path(self, turn_intent=None):
        self.set_turn_intent(turn_intent)
        self.route.update(max(self.cfg.CONTROL_LOOKAHEAD_M,
                              self.cfg.EVADE_LOOKAHEAD_M+15., 40.), self._turn_intent)
        self._route_prepared = True

    def _planned_path(self):
        cfg = self.cfg
        if not self._route_prepared:
            self.prepare_path(self._turn_intent)
        self._route_prepared = False
        path = self.route.path(lookahead_m=cfg.CONTROL_LOOKAHEAD_M)
        context = self.route.context
        if not path or not context['route_valid']:
            raise RuntimeError(context['route_reason'] or 'unresolved map route')
        current_lane = context.get("current_lane_id")
        if (self._lane_change_side is not None and self._lane_change_origin is not None
                and current_lane != self._lane_change_origin):
            self._lane_change_side = None
            self._lane_change_origin = None
        if self._lane_change_side is not None:
            offset = 2.0 * getattr(cfg, 'LANE_HALF_WIDTH_M', 1.75)
            if self._lane_change_side == "left":
                offset = -offset
            path = self.local.lane_change_trajectory(
                path, offset, cfg.LANE_CHANGE_TRANSITION_M)
        return path

    def run_step(self, target_speed_kmh, nearest_distance=None,
                 traffic_light_state="green", stop_sign=False,
                 respect_traffic_controls=True):
        speed_ms = self._speed_ms(self.ego)
        if not respect_traffic_controls:
            traffic_light_state, stop_sign = "green", False
        stop_pending = self.stop_sign.update(
            stop_sign, speed_ms, self.cfg.FIXED_DELTA)
        # CARLA signal state là safety fallback nếu vision chưa chắc chắn.
        try:
            if respect_traffic_controls and self.ego.is_at_traffic_light():
                carla_state = str(self.ego.get_traffic_light_state()).split('.')[-1].lower()
                if carla_state in ("red", "yellow"):
                    traffic_light_state = carla_state
        except Exception:
            pass
        # No collision_risk key: the behaviour planner is tactical only, and
        # braking is arbitrated in ego_control.py. Passing a hardcoded False
        # here is what made the planner's old emergency branch unreachable.
        metrics = {
            "nearest_distance": 99.0 if nearest_distance is None else nearest_distance,
            "traffic_light": str(traffic_light_state).capitalize(),
        }
        state = self.behavior.update_state(metrics)
        target_kmh = max(0.0, float(target_speed_kmh))
        signal_stop = str(traffic_light_state).lower() in ("red", "yellow")
        if state == DrivingState.STOPPING_AT_LIGHT or signal_stop or stop_pending:
            target_kmh = 0.0
        elif state == DrivingState.FOLLOWING_VEHICLE:
            target_kmh = min(target_kmh, speed_ms * 3.6)

        path = self._planned_path()
        steer = self.lateral.compute_path_steering(
            path, speed_ms, self.cfg.CONTROL_WHEELBASE_M,
            self.cfg.CONTROL_MAX_STEER_DEG)
        throttle, brake = self.longitudinal.update(
            target_kmh / 3.6, speed_ms, self.cfg.FIXED_DELTA)
        control = self.vehicle_control.create_control_command(throttle, brake, steer)
        return control, {
            "mode": "custom",
            "behavior_state": state.name,
            "target_speed_kmh": round(target_kmh, 2),
            "lane_change_side": self._lane_change_side,
            "turn_intent": self._turn_intent,
            "stop_sign_pending": stop_pending,
            "traffic_light_state": str(traffic_light_state).lower(),
            "path_points": len(path),
            "route_valid": self.route.context['route_valid'],
            "route_road_id": self.route.context.get('road_id'),
        }
