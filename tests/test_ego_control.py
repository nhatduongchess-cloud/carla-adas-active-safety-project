"""Offline regression tests for ego control ownership and custom-fault handling."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from modules.ego_control import EgoController


def config(mode):
    return SimpleNamespace(
        EGO_CONTROL_MODE=mode,
        LANE_CHANGE_COOLDOWN_S=8.0,
        FIXED_DELTA=0.025,
        TM_DEFAULT_SPEED_DIFF=0.0,
    )


def normal_decision():
    return SimpleNamespace(action="DRIVE", state="NORMAL", slow_pct=0.0, threat={})


class EgoControllerRegressionTests(unittest.TestCase):
    @patch("modules.ego_control.set_hazard_lights")
    @patch("modules.ego_control.carla.VehicleControl", side_effect=lambda **kw: kw)
    @patch("modules.ego_control.EgoDrivingStack")
    def test_custom_fault_latches_manual_safe_stop_without_tm(
            self, stack_cls, _control, _hazard):
        ego = MagicMock()
        tm = MagicMock()
        stack_cls.return_value.run_step.side_effect = RuntimeError("PID failed")
        controller = EgoController(ego, tm, config("custom"), world=MagicMock())

        status = controller.apply(normal_decision(), {"override": False, "hazard": False},
                                  "NORMAL", 1, target_speed_kmh=20.0)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assertEqual(status["behavior_state"], "CUSTOM_CONTROL_FAULT_SAFE_STOP")
        self.assertEqual(ego.apply_control.call_args.args[0]["brake"], 1.0)
        self.assertFalse(any(call.args and call.args[0] is True
                             for call in ego.set_autopilot.call_args_list))
        tm.get_port.assert_not_called()

        # The failure remains latched on subsequent frames; it cannot fall through to TM.
        controller.apply(normal_decision(), {"override": False, "hazard": False},
                         "NORMAL", 2, target_speed_kmh=20.0)
        self.assertEqual(controller.mode, "custom_fault_safe_stop")
        tm.get_port.assert_not_called()

    @patch("modules.ego_control.set_hazard_lights")
    def test_traffic_manager_is_enabled_only_for_explicit_mode(self, _hazard):
        ego = MagicMock()
        tm = MagicMock()
        tm.get_port.return_value = 8000
        controller = EgoController(ego, tm, config("traffic_manager"), world=MagicMock())

        ego.set_autopilot.assert_not_called()
        status = controller.apply(normal_decision(), {"override": False, "hazard": False},
                                  "NORMAL", 1)
        self.assertEqual(status["mode"], "traffic_manager")
        ego.set_autopilot.assert_called_once_with(True, 8000)


class EgoControllerFaultBoundaryTests(unittest.TestCase):
    def setUp(self):
        stack_patch = patch("modules.ego_control.EgoDrivingStack")
        self.stack_cls = stack_patch.start()
        self.addCleanup(stack_patch.stop)
        self.ego = MagicMock()
        self.tm = MagicMock()
        self.world = MagicMock()
        self.l3 = {"override": False, "hazard": False}

    def controller(self):
        return EgoController(self.ego, self.tm, config("custom"), world=self.world)

    def fail_controller(self, error=None):
        self.stack_cls.return_value.run_step.side_effect = (
            error if error is not None else RuntimeError("PID failed"))
        controller = self.controller()
        controller.apply(normal_decision(), self.l3, "NORMAL", 1, target_speed_kmh=20.0)
        return controller

    def assert_full_brake(self, hand_brake=False):
        control = self.ego.apply_control.call_args.args[0]
        self.assertEqual(control.brake, 1.0)
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.hand_brake, hand_brake)
        self.tm.get_port.assert_not_called()
        self.assertFalse(any(call.args and call.args[0] is True
                             for call in self.ego.set_autopilot.call_args_list))

    def test_custom_requires_world_before_any_actor_or_stack_call(self):
        with self.assertRaises(ValueError):
            EgoController(self.ego, self.tm, config("custom"), world=None)
        self.stack_cls.assert_not_called()
        self.assertEqual(self.ego.mock_calls, [])
        self.assertEqual(self.tm.mock_calls, [])

    def test_invalid_or_missing_mode_fails_closed(self):
        for mode in (None, "", "autopilot", "custmo", 42):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                EgoController(self.ego, self.tm, config(mode), world=self.world)
        cfg = config("custom")
        del cfg.EGO_CONTROL_MODE
        with self.assertRaises(ValueError):
            EgoController(self.ego, self.tm, cfg, world=self.world)
        self.stack_cls.assert_not_called()
        self.assertEqual(self.ego.mock_calls, [])
        self.assertEqual(self.tm.mock_calls, [])

    def test_explicit_tm_requires_manager(self):
        with self.assertRaises(ValueError):
            EgoController(self.ego, None, config("traffic_manager"), world=self.world)
        self.assertEqual(self.ego.mock_calls, [])

    def test_explicit_tm_can_initialize_without_world(self):
        controller = EgoController(self.ego, self.tm, config("traffic_manager"))
        self.assertTrue(controller.uses_traffic_manager_control)
        self.stack_cls.assert_not_called()
        self.ego.set_autopilot.assert_not_called()

    def test_unicode_log_failure_cannot_prevent_braking(self):
        error = UnicodeEncodeError("cp1252", "\u0111", 0, 1, "unsupported")
        self._check_broken_log(error)

    def test_broken_console_cannot_prevent_braking(self):
        self._check_broken_log(BrokenPipeError("console closed"))

    def _check_broken_log(self, error):
        controls_at_log = []
        def failing_log(*args, **kwargs):
            controls_at_log.append(self.ego.apply_control.call_args)
            raise error
        with patch("builtins.print", side_effect=failing_log) as log:
            controller = self.fail_controller()
            status = controller.apply(normal_decision(), self.l3, "NORMAL", 2)
        self.assertEqual(log.call_count, 1)
        self.assertIsNotNone(controls_at_log[0])
        self.assertEqual(controls_at_log[0].args[0].brake, 1.0)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assert_full_brake()

    def test_error_with_broken_string_conversion_does_not_break_safe_stop(self):
        class UnprintableError(RuntimeError):
            def __str__(self):
                raise ValueError("broken error formatter")
        controller = self.fail_controller(UnprintableError())
        status = controller.apply(normal_decision(), self.l3, "NORMAL", 2)
        self.assertIn("UnprintableError", status["control_error"])
        self.assert_full_brake()

    def test_hazard_light_rpc_failure_does_not_prevent_braking(self):
        self.ego.set_light_state.side_effect = RuntimeError("lights unavailable")
        controller = self.fail_controller()
        status = controller.apply(normal_decision(), self.l3, "NORMAL", 2)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assert_full_brake()

    def test_aeb_cannot_weaken_latched_fault_braking(self):
        controller = self.fail_controller()
        decision = SimpleNamespace(action="BRAKE", brake=0.3, state="BRAKE_TO_STOP")
        status = controller.apply(decision, self.l3, "NORMAL", 2)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assert_full_brake()
        self.stack_cls.return_value.run_step.assert_called_once()

    def test_mrm_cannot_weaken_latched_fault_braking(self):
        controller = self.fail_controller()
        l3 = {"override": True, "state": "MRM", "target_decel_ms2": 0.6, "hazard": True}
        status = controller.apply(normal_decision(), l3, "VIOLATION", 2)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assert_full_brake()

    def test_safe_stop_hold_is_preserved_in_fault_latch(self):
        controller = self.fail_controller()
        l3 = {"override": True, "state": "SAFE_STOP", "hazard": True}
        controller.apply(normal_decision(), l3, "VIOLATION", 2)
        self.assert_full_brake(hand_brake=True)
        controller.apply(normal_decision(), self.l3, "NORMAL", 3)
        self.assert_full_brake(hand_brake=True)

    def test_failed_brake_rpc_propagates_but_retains_fault_latch(self):
        self.stack_cls.return_value.run_step.side_effect = RuntimeError("PID failed")
        controller = self.controller()
        self.ego.apply_control.side_effect = RuntimeError("server disconnected")
        with self.assertRaisesRegex(RuntimeError, "server disconnected"):
            controller.apply(normal_decision(), self.l3, "NORMAL", 1, target_speed_kmh=20)
        self.assertEqual(controller.mode, "custom_fault_safe_stop")
        self.assertIsNone(controller.custom)
        self.ego.apply_control.side_effect = None
        status = controller.apply(normal_decision(), self.l3, "NORMAL", 2)
        self.assertEqual(status["mode"], "custom_fault_safe_stop")
        self.assert_full_brake()

    def test_healthy_custom_command_is_unchanged(self):
        # A valid command (the fixture was a bare object(); commands are now
        # validated before the RPC, so the fixture has to be a command).
        control = SimpleNamespace(throttle=0.4, steer=0.1, brake=0.0)
        expected_status = {"mode": "custom", "behavior_state": "CRUISE"}
        self.stack_cls.return_value.run_step.return_value = (control, expected_status)
        controller = self.controller()
        status = controller.apply(normal_decision(), self.l3, "NORMAL", 1, target_speed_kmh=20)
        self.assertEqual(status, expected_status)
        self.ego.apply_control.assert_called_once_with(control)

    def test_healthy_aeb_and_l3_priority_are_unchanged(self):
        controller = self.controller()
        decision = SimpleNamespace(action="BRAKE", brake=0.3, state="BRAKE_TO_STOP")
        status = controller.apply(decision, self.l3, "NORMAL", 1)
        self.assertEqual(status["mode"], "aeb_override")
        self.assertAlmostEqual(self.ego.apply_control.call_args.args[0].brake, 0.3)
        l3 = {"override": True, "state": "MRM", "target_decel_ms2": 3.6, "hazard": True}
        status = controller.apply(decision, l3, "VIOLATION", 2)
        self.assertEqual(status["mode"], "l3_override")
        self.assertAlmostEqual(self.ego.apply_control.call_args.args[0].brake, 0.6)
        self.stack_cls.return_value.run_step.assert_not_called()


class EgoControllerEvidenceRemediationTests(unittest.TestCase):
    """WP01 of docs/CLAUDE_CARLA_EVIDENCE_REMEDIATION_20260925.md."""

    # Borrow the fixtures rather than subclass, so the parent's tests are not
    # collected and counted a second time.
    setUp = EgoControllerFaultBoundaryTests.setUp
    controller = EgoControllerFaultBoundaryTests.controller
    assert_full_brake = EgoControllerFaultBoundaryTests.assert_full_brake

    MRM = {"override": True, "state": "MRM_EXECUTING", "target_decel_ms2": 1.0, "hazard": True}

    def sent(self):
        return self.ego.apply_control.call_args.args[0]

    def test_aeb_full_brake_during_mrm_reaches_the_vehicle(self):
        controller = self.controller()
        decision = SimpleNamespace(action="BRAKE", brake=1.0, state="EMERGENCY_BRAKE")
        status = controller.apply(decision, self.MRM, "VIOLATION", 1)
        self.assertEqual((self.sent().brake, self.sent().throttle), (1.0, 0.0))
        self.assertEqual(status["brake_source"], "aeb")
        self.assertEqual(status["requested"]["aeb_brake"], 1.0)
        self.assertEqual(self.ego.apply_control.call_count, 1)

    def test_stronger_mrm_request_wins_over_weaker_aeb(self):
        controller = self.controller()
        mrm = dict(self.MRM, target_decel_ms2=4.2)          # 0.7 on the pedal
        decision = SimpleNamespace(action="BRAKE", brake=0.4, state="BRAKE_HOLD")
        controller.apply(decision, mrm, "VIOLATION", 1)
        self.assertAlmostEqual(self.sent().brake, 0.7)

    def test_safe_stop_holds_and_never_engages_autopilot(self):
        controller = self.controller()
        controller.apply(normal_decision(), {"override": True, "state": "SAFE_STOP",
                                             "hazard": True}, "VIOLATION", 1)
        self.assertEqual((self.sent().brake, self.sent().hand_brake), (1.0, True))
        self.assertFalse(any(call.args and call.args[0] is True
                             for call in self.ego.set_autopilot.call_args_list))

    def test_non_finite_aeb_request_becomes_full_brake_not_nan(self):
        controller = self.controller()
        decision = SimpleNamespace(action="BRAKE", brake=float("nan"), state="EMERGENCY_BRAKE")
        controller.apply(decision, self.l3, "NORMAL", 1)
        self.assertEqual(self.sent().brake, 1.0)

    def test_invalid_custom_command_never_reaches_the_rpc(self):
        for bad in ({"throttle": float("nan")}, {"steer": 3.0}, {"brake": -0.1},
                    {"throttle": True}):
            with self.subTest(bad=bad):
                self.ego.reset_mock()
                fields = dict(throttle=0.3, steer=0.0, brake=0.0, **bad)
                control = SimpleNamespace(**fields)
                self.stack_cls.return_value.run_step.side_effect = None
                self.stack_cls.return_value.run_step.return_value = (control, {"mode": "custom"})
                controller = self.controller()
                status = controller.apply(normal_decision(), self.l3, "NORMAL", 1,
                                          target_speed_kmh=20)
                self.assertEqual(status["mode"], "custom_fault_safe_stop")
                self.assertIn("invalid control command", status["control_error"])
                self.assertIsNot(self.sent(), control)
                self.assert_full_brake()

    def test_brake_and_throttle_are_never_sent_together(self):
        control = SimpleNamespace(throttle=0.5, steer=0.0, brake=0.2)
        self.stack_cls.return_value.run_step.return_value = (control, {"mode": "custom"})
        self.controller().apply(normal_decision(), self.l3, "NORMAL", 1, target_speed_kmh=20)
        self.assertEqual((self.sent().throttle, self.sent().brake), (0.0, 0.2))

    def test_sensor_loss_safe_stop_reports_an_rpc_failure_as_a_failure(self):
        controller = self.controller()
        self.ego.apply_control.side_effect = RuntimeError("server disconnected")
        outcome = controller.sensor_loss_safe_stop("lidar timeout")
        self.assertEqual(controller.mode, "custom_fault_safe_stop")
        self.assertTrue(outcome["attempted"])
        self.assertFalse(outcome["command_sent"])
        self.assertIn("server disconnected", outcome["error"])

    def test_sensor_loss_safe_stop_sends_full_brake_when_the_server_answers(self):
        controller = self.controller()
        outcome = controller.sensor_loss_safe_stop("lidar timeout")
        self.assertTrue(outcome["command_sent"])
        self.assert_full_brake()


if __name__ == "__main__":
    unittest.main()
