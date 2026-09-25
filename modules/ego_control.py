"""Bộ điều khiển ego dùng chung: dịch quyết định an toàn + L3 thành lệnh CARLA.

Tách ra để chinh.py / evaluate_l3.py / run_scenarios.py không lặp lại logic. Ưu
tiên: latched custom fault > L3 override (MRM) > AEB (BRAKE) > custom control.
Inside the L3 override the brake is max(MRM, AEB), never MRM alone. Every
command is validated before the RPC; throttle is zero whenever brake > 0.
Fault latch vẫn giữ hand brake khi L3 yêu cầu SAFE_STOP. Traffic Manager chỉ là
chế độ opt-in; custom-control fault phải safe-stop, không được tự bật autopilot.
"""
# fmt: off
# isort: skip_file
import carla

try:
    from control_arbitration import (aeb_brake, command_problems, exclusive_pedals,
                                     longitudinal_override)
    from ego_driving_stack import EgoDrivingStack
    from road_geometry import EgoRoute
except ImportError:
    from modules.control_arbitration import (aeb_brake, command_problems, exclusive_pedals,
                                             longitudinal_override)
    from modules.ego_driving_stack import EgoDrivingStack
    from modules.road_geometry import EgoRoute


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
    def __init__(self, ego, traffic_manager, cfg, world=None):
        requested = getattr(cfg, 'EGO_CONTROL_MODE', None)
        if not isinstance(requested, str) or requested.lower() not in (
                'custom', 'traffic_manager'):
            raise ValueError("EGO_CONTROL_MODE must explicitly select custom or traffic_manager")
        requested = requested.lower()
        if requested == 'custom' and world is None:
            raise ValueError("custom ego control requires world; implicit TM fallback is forbidden")
        if requested == 'traffic_manager' and traffic_manager is None:
            raise ValueError("traffic_manager ego control requires a Traffic Manager")
        self.ego = ego
        self.tm = traffic_manager
        self.cfg = cfg
        # chinh.py không đăng ký ego với TM lúc spawn. Với chế độ TM opt-in,
        # _ensure_tm() là điểm duy nhất được phép bật autopilot.
        self.engaged = False
        self.last_lane_change_frame = -(10 ** 9)
        self.cooldown_frames = cfg.LANE_CHANGE_COOLDOWN_S / cfg.FIXED_DELTA
        self.mode = requested
        self.route = EgoRoute(world, ego) if world is not None else None
        self.custom = EgoDrivingStack(world, ego, cfg, route=self.route) if self.mode == "custom" else None
        self._fallback_reported = False
        self._custom_fault_error = None
        self._fault_hand_brake = False
        if self.mode == "custom":
            ego.set_autopilot(False)
            self.engaged = False

    @property
    def uses_custom_control(self):
        return self.mode == "custom" and self.custom is not None

    @property
    def uses_traffic_manager_control(self):
        """True chỉ khi người dùng đã chủ động chọn TM cho ego."""
        return self.mode == "traffic_manager"

    def path_context(self, turn_intent=None):
        """Prepare one map reference before safety; custom control consumes it."""
        if self.uses_custom_control:
            self.custom.prepare_path(turn_intent)
        elif self.route is not None:
            self.route.update(max(self.cfg.CONTROL_LOOKAHEAD_M,
                                  self.cfg.EVADE_LOOKAHEAD_M+15., 40.), turn_intent)
        else:
            raise RuntimeError('map route requires world')
        return (self.route.path(self.cfg.LIDAR_X, max(self.cfg.EVADE_LOOKAHEAD_M+15., 40.)),
                dict(self.route.context))

    def _ensure_tm(self):
        if not self.engaged:
            self.ego.set_autopilot(True, self.tm.get_port())
            self.engaged = True

    def _ensure_manual(self):
        if self.engaged:
            self.ego.set_autopilot(False)
            self.engaged = False

    def _apply_tm(self, decision, odd_state, frame):
        ego, tm, cfg = self.ego, self.tm, self.cfg
        self._ensure_tm()
        drive_diff = 40.0 if odd_state == "DEGRADED" else cfg.TM_DEFAULT_SPEED_DIFF
        if decision.action == "SLOW":
            tm.vehicle_percentage_speed_difference(
                ego, max(decision.slow_pct, drive_diff))
        elif decision.action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            tm.vehicle_percentage_speed_difference(ego, drive_diff)
            if frame - self.last_lane_change_frame > self.cooldown_frames:
                tm.force_lane_change(ego, decision.action == "LANE_CHANGE_RIGHT")
                self.last_lane_change_frame = frame
        else:
            tm.vehicle_percentage_speed_difference(ego, drive_diff)
        return {"mode": "traffic_manager", "behavior_state": decision.state}

    def _apply_custom_fault_safe_stop(self, error=None, hand_brake=False):
        """Latch a manual safe stop after a custom-control failure.

        This path deliberately never calls the CARLA Traffic Manager.  It keeps
        a faulty perception/control stack fail-safe and avoids silently routing
        the ego into the independently unstable autopilot integration.
        """
        self._fault_hand_brake = self._fault_hand_brake or bool(hand_brake)
        self._ensure_manual()
        # Control RPC failures must propagate: an attempted stop is not an
        # acknowledged stop when the server has disconnected.
        self.ego.apply_control(carla.VehicleControl(
            brake=1.0, hand_brake=self._fault_hand_brake))
        set_hazard_lights(self.ego, True)
        status = {
            "mode": "custom_fault_safe_stop",
            "behavior_state": "CUSTOM_CONTROL_FAULT_SAFE_STOP",
        }
        if error is not None:
            try:
                self._custom_fault_error = str(error)
            except Exception:
                self._custom_fault_error = type(error).__name__ + " (message unavailable)"
        if self._custom_fault_error is not None:
            status["control_error"] = self._custom_fault_error
        return status

    def sensor_loss_safe_stop(self, reason):
        """Latch the fault safe stop because a required sensor stopped delivering.

        Returns the attempt outcome. A raised RPC error is recorded, not
        swallowed into "stopped": if the server is gone, no brake was applied.
        """
        self.mode = "custom_fault_safe_stop"
        self.custom = None
        outcome = {"reason": str(reason), "attempted": True, "command_sent": False,
                   "error": None}
        try:
            self._apply_custom_fault_safe_stop(error=reason)
            outcome["command_sent"] = True
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        return outcome

    def apply(self, decision, l3, odd_state, frame, target_speed_kmh=None,
              traffic_control=None, turn_intent=None,
              respect_traffic_controls=True):
        ego = self.ego

        if self.mode == "custom_fault_safe_stop":
            return self._apply_custom_fault_safe_stop(
                hand_brake=bool(l3.get('override') and l3.get('state') == 'SAFE_STOP'))

        if l3['override']:
            # The MRM owns the vehicle, but it does not get to veto a harder
            # AEB brake: see modules/control_arbitration.py.
            self._ensure_manual()
            command = longitudinal_override(l3, decision)
            # Steer is held at 0 during an MRM/AEB override. That is an explicit
            # degraded lateral fallback, not lane keeping: on a curve the car
            # leaves the lane (open gap G10 in docs/ARCHITECTURE.md).
            ego.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0,
                brake=command["brake"], hand_brake=command["hand_brake"]))
            set_hazard_lights(ego, True)
            status = {"mode": "l3_override", "behavior_state": l3['state'],
                      "brake_source": command["source"],
                      "requested": {"aeb_brake": command["aeb_brake"],
                                    "mrm_brake": command["mrm_brake"]},
                      "applied": {"throttle": 0.0, "steer": 0.0,
                                  "brake": command["brake"],
                                  "hand_brake": command["hand_brake"]},
                      # apply_control is fire-and-forget: sent, not acknowledged.
                      "command_sent": True}
            return status

        elif decision.action == "BRAKE":
            set_hazard_lights(ego, l3['hazard'])
            self._ensure_manual()
            brake = aeb_brake(decision)
            ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake))
            return {"mode": "aeb_override", "behavior_state": decision.state,
                    "requested": {"aeb_brake": getattr(decision, "brake", None)},
                    "applied": {"throttle": 0.0, "steer": 0.0, "brake": brake},
                    "command_sent": True}

        set_hazard_lights(ego, l3['hazard'])
        if not self.uses_custom_control:
            return self._apply_tm(decision, odd_state, frame)

        try:
            self._ensure_manual()
            self.custom.set_turn_intent(turn_intent)
            target = float(target_speed_kmh if target_speed_kmh is not None
                           else max(10.0, ego.get_speed_limit()))
            if odd_state == "DEGRADED":
                target = min(target, 40.0)
            if decision.action == "SLOW":
                target *= max(0.0, 1.0 - decision.slow_pct / 100.0)
            elif decision.action in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
                side = "right" if decision.action.endswith("RIGHT") else "left"
                if not self.custom.request_lane_change(side):
                    # Safety FSM đã yêu cầu né nhưng topology runtime từ chối.
                    ego.apply_control(carla.VehicleControl(brake=1.0))
                    return {"mode": "custom_abort", "behavior_state": "LANE_CHANGE_REJECTED"}

            signal = traffic_control or {}
            control, status = self.custom.run_step(
                target,
                nearest_distance=(decision.threat or {}).get("distance_m"),
                traffic_light_state=signal.get("traffic_light_state", "green"),
                stop_sign=bool(signal.get("stop_sign", False)),
                respect_traffic_controls=respect_traffic_controls)
            # Validate before the RPC: a NaN or out-of-range command from the
            # stack is a control fault, handled by the safe stop below, never
            # silently clamped and sent.
            problems = command_problems(throttle=control.throttle, steer=control.steer,
                                        brake=control.brake)
            if problems:
                raise ValueError("invalid control command: " + "; ".join(problems))
            control.throttle = exclusive_pedals(control.throttle, control.brake)
            ego.apply_control(control)
            return status
        except Exception as exc:
            # Latch first, brake second, diagnose last. Neither error formatting
            # nor an unavailable Windows console may prevent the safety command.
            self.mode = "custom_fault_safe_stop"
            self.custom = None
            status = self._apply_custom_fault_safe_stop(exc)
            if not self._fallback_reported:
                self._fallback_reported = True
                try:
                    print(f"[Control] Custom stack error ({status['control_error']}) "
                          "-> manual safe stop; TM ego fallback disabled.")
                except Exception:
                    # The fault remains available in returned status even if
                    # stdout is closed or cannot encode its message.
                    pass
            return status
