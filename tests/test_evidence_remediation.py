"""Regression tests for docs/CLAUDE_CARLA_EVIDENCE_REMEDIATION_20260925.md.

`tests/test_evidence_review.py` covers the fourteen probes of the review
script. This file covers the work packages of the written brief that go
beyond those probes: command validation (WP01), input contracts, range
health and evidence-based brake release (WP02), takeover ownership (WP03),
case/suite statuses (WP05), and control-rate measurement (WP06).

Every test runs without CARLA or a simulator. Where a test says "simulation"
it means a pure-Python model of a code path, not a vehicle run.
"""

from __future__ import annotations

import math
import unittest

from modules.active_safety import ActiveSafetySystem
from modules.control_arbitration import command_problems, exclusive_pedals
from modules.control_timing import ControlWindow, SimClock
from modules.kpi import KpiRecorder
from modules.l3_report import (assess_profile, build_report, case_id, exit_code_for,
                               write_report)
from modules.mrm_controller import L3StateMachine
from modules.odd_monitor import ODDMonitor
from modules.scenario_acceptance import assess_scenario, finalize_after_cleanup, reaction_timing
from modules.sensor_health import SensorHealthMonitor
from scripts.tools import generate_validation_report as gvr

NAN = math.nan
CLEAR = {"visibility_m": 200.0, "mu": 0.9, "snr": 1.0}


# ---------------------------------------------------------------------------
# WP01 - a command is validated before it can reach the simulator.
# ---------------------------------------------------------------------------

class CommandValidationTests(unittest.TestCase):
    def test_a_valid_command_has_no_problems(self):
        self.assertEqual(command_problems(throttle=0.4, steer=-0.2, brake=0.0), [])

    def test_every_invalid_kind_is_named(self):
        cases = {
            "throttle": [NAN, math.inf, -0.01, 1.01, True, "0.5", None],
            "brake": [NAN, -math.inf, -0.1, 2.0, False],
            "steer": [NAN, -1.01, 1.01, "left"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    problems = command_problems(**{field: value})
                    self.assertEqual(len(problems), 1)
                    self.assertIn(field, problems[0])

    def test_numpy_scalars_are_numbers(self):
        import numpy as np
        self.assertEqual(command_problems(throttle=np.float32(0.3), brake=np.float64(0.0)), [])

    def test_brake_always_zeroes_the_throttle(self):
        self.assertEqual(exclusive_pedals(0.6, 0.01), 0.0)
        self.assertEqual(exclusive_pedals(0.6, 0.0), 0.6)


# ---------------------------------------------------------------------------
# WP02 - input contract of the ODD monitor.
# ---------------------------------------------------------------------------

class OddInputContractTests(unittest.TestCase):
    def classify(self, **overrides):
        conditions = dict(CLEAR, **overrides)
        return ODDMonitor().classify(conditions)

    def test_every_invalid_input_kind_is_a_violation_never_normal(self):
        for key in ("visibility_m", "mu", "snr"):
            for bad in (NAN, math.inf, -math.inf, None, True, False, "0.9", [1.0]):
                with self.subTest(key=key, value=bad):
                    r = self.classify(**{key: bad})
                    self.assertEqual(r["state"], "VIOLATION")
                    self.assertIn(key, r["reason"])

    def test_a_missing_key_is_unmeasured_not_perfect(self):
        for key in ("visibility_m", "mu", "snr"):
            with self.subTest(missing=key):
                conditions = {k: v for k, v in CLEAR.items() if k != key}
                r = ODDMonitor().classify(conditions)
                self.assertEqual(r["state"], "VIOLATION")
                self.assertEqual(r["reason"], f"odd_unmeasurable:{key}")

    def test_unmeasurable_asks_for_takeover_rather_than_skipping_to_mrm(self):
        self.assertFalse(self.classify(visibility_m=NAN)["critical"])

    def test_published_boundaries_are_unchanged(self):
        expected = [
            ({"visibility_m": 19.9}, "VIOLATION"),
            ({"visibility_m": 20.0}, "DEGRADED"),
            ({"visibility_m": 50.0}, "DEGRADED"),
            ({"visibility_m": 50.1}, "NORMAL"),
            ({"mu": 0.299}, "VIOLATION"),
            ({"mu": 0.3}, "DEGRADED"),
            ({"mu": 0.599}, "DEGRADED"),
            ({"mu": 0.6}, "NORMAL"),
            ({"snr": 0.249}, "VIOLATION"),
            ({"snr": 0.25}, "NORMAL"),
        ]
        for overrides, state in expected:
            with self.subTest(**overrides):
                self.assertEqual(self.classify(**overrides)["state"], state)

    def test_critical_is_an_absolute_threshold(self):
        # Injected-condition tests: the weather model cannot produce mu < 0.3
        # (see G11); these exercise the branch directly.
        self.assertTrue(self.classify(visibility_m=9.99)["critical"])
        self.assertFalse(self.classify(visibility_m=10.0)["critical"])
        self.assertTrue(self.classify(mu=0.19)["critical"])
        self.assertFalse(self.classify(mu=0.2)["critical"])


# ---------------------------------------------------------------------------
# WP02 - range availability per configured sensor.
# ---------------------------------------------------------------------------

class RangeHealthTests(unittest.TestCase):
    def run_frames(self, enabled, lost, frames=5):
        health = SensorHealthMonitor(enabled)
        for i in range(frames):
            health.next_frame()
            for sensor in ("camera", "lidar", "radar"):
                health.observe(sensor, i, i * 0.025, sensor not in lost)
        return health.summary()

    def test_the_matrix(self):
        both = {"camera": True, "lidar": True, "radar": True}
        no_radar = {"camera": True, "lidar": True, "radar": False}
        cases = [
            (both, (), False, False),
            (both, ("lidar",), False, True),
            (both, ("radar",), False, True),
            (both, ("lidar", "radar"), True, False),
            (no_radar, (), False, False),
            (no_radar, ("lidar",), True, False),
        ]
        for enabled, lost, all_lost, degraded in cases:
            with self.subTest(radar=enabled["radar"], lost=lost):
                summary = self.run_frames(enabled, lost)
                self.assertEqual(summary["all_enabled_range_unavailable"], all_lost)
                self.assertEqual(summary["range_redundancy_lost"], all_lost)
                self.assertEqual(summary["range_redundancy_degraded"], degraded)

    def test_total_range_loss_is_an_odd_violation_and_critical(self):
        summary = self.run_frames({"camera": True, "lidar": True, "radar": False}, ("lidar",))
        r = ODDMonitor().classify(dict(CLEAR), summary)
        self.assertEqual(r["state"], "VIOLATION")
        self.assertTrue(r["critical"])
        self.assertEqual(r["reason"], "all_enabled_range_sensors_unavailable")


# ---------------------------------------------------------------------------
# WP02 - a latched brake is released by evidence, not by missing data.
# ---------------------------------------------------------------------------

class BrakeReleaseEvidenceTests(unittest.TestCase):
    OBSTACLE = {"centroid": (4.0, 0.0, 0.0), "point_count": 10}

    def latched(self):
        safety = ActiveSafetySystem(enable_evasion=False)
        d = safety.update(1.0, [], [self.OBSTACLE], 0.025)
        self.assertEqual((d.action, d.brake), ("BRAKE", 1.0))
        return safety

    def test_missing_lidar_never_releases_and_never_weakens(self):
        safety = self.latched()
        for _ in range(400):                       # 10 s at 40 Hz
            d = safety.update(0.0, [], [], 0.025, lidar_valid=False)
            self.assertEqual((d.action, d.brake), ("BRAKE", 1.0))
        self.assertEqual(d.state, "BRAKE_HOLD_NO_DATA")

    def test_observed_clear_path_releases_after_the_confirmation_time(self):
        safety = self.latched()
        actions = [safety.update(0.0, [], [], 0.025).action for _ in range(5)]
        # Same cadence as before at 40 Hz: four holds, release on the fifth.
        self.assertEqual(actions, ["BRAKE"] * 4 + ["DRIVE"])
        self.assertIn("release", safety.last_transition_reason)

    def test_hold_does_not_drop_from_full_brake(self):
        safety = self.latched()
        d = safety.update(0.0, [], [], 0.025)
        self.assertEqual((d.state, d.brake), ("BRAKE_HOLD", 1.0))

    def test_release_is_by_time_not_frame_count(self):
        safety = ActiveSafetySystem(enable_evasion=False)
        safety.update(1.0, [], [self.OBSTACLE], 0.1)            # 10 Hz loop
        actions = [safety.update(0.0, [], [], 0.1).action for _ in range(2)]
        self.assertEqual(actions, ["BRAKE", "DRIVE"])

    def test_a_data_gap_restarts_the_confirmation(self):
        safety = self.latched()
        seq = [safety.update(0.0, [], [], 0.025).action for _ in range(3)]
        seq.append(safety.update(0.0, [], [], 0.025, lidar_valid=False).action)
        seq += [safety.update(0.0, [], [], 0.025).action for _ in range(5)]
        self.assertEqual(seq, ["BRAKE"] * 8 + ["DRIVE"])

    def test_invalid_dt_is_not_evidence_of_elapsed_clear_time(self):
        safety = self.latched()
        for bad_dt in (NAN, -0.025, 0.0, math.inf):
            d = safety.update(0.0, [], [], bad_dt)
            self.assertEqual(d.action, "BRAKE", bad_dt)

    def test_the_brake_is_not_permanent_once_the_path_is_seen_clear(self):
        safety = self.latched()
        for _ in range(40):
            safety.update(0.0, [], [], 0.025, lidar_valid=False)
        actions = [safety.update(0.0, [], [], 0.025).action for _ in range(5)]
        self.assertEqual(actions[-1], "DRIVE")



# ---------------------------------------------------------------------------
# WP03 - takeover, ownership and event counting.
# ---------------------------------------------------------------------------

class TakeoverOwnershipTests(unittest.TestCase):
    def machine_in_tor(self):
        m = L3StateMachine()
        r = m.update("VIOLATION", False, 10.0, 0.4, 0.025)
        self.assertEqual(r["state"], "TAKEOVER_REQUEST")
        return m

    def test_normal_degraded_normal(self):
        m = L3StateMachine()
        states = [m.update(odd, False, 10.0, 0.8, 0.025)["state"]
                  for odd in ("NORMAL", "DEGRADED", "NORMAL")]
        self.assertEqual(states, ["L3_ACTIVE", "DEGRADED", "L3_ACTIVE"])

    def test_full_fallback_chain(self):
        m = self.machine_in_tor()
        for _ in range(399):
            r = m.update("VIOLATION", False, 10.0, 0.4, 0.025)
        self.assertEqual(r["state"], "TAKEOVER_REQUEST")        # 9.975 s
        r = m.update("VIOLATION", False, 10.0, 0.4, 0.025)
        self.assertEqual(r["state"], "MRM_EXECUTING")           # 10.0 s
        self.assertTrue(r["override"])
        r = m.update("VIOLATION", False, 0.2, 0.4, 0.025)
        self.assertEqual(r["state"], "SAFE_STOP")

    def test_uneven_dt_uses_elapsed_time(self):
        m = self.machine_in_tor()
        for dt in (4.0, 5.99):
            self.assertEqual(m.update("VIOLATION", False, 10., .4, dt)["state"],
                             "TAKEOVER_REQUEST")
        self.assertEqual(m.update("VIOLATION", False, 10., .4, 0.02)["state"], "MRM_EXECUTING")

    def test_critical_skips_the_tor(self):
        m = L3StateMachine()
        r = m.update("VIOLATION", False, 10.0, 0.1, 0.025, critical=True)
        self.assertEqual(r["state"], "MRM_EXECUTING")

    def test_ack_in_violation_never_reengages_or_takes_the_wheel_back(self):
        m = self.machine_in_tor()
        r = m.update("VIOLATION", True, 10.0, 0.4, 0.025)
        self.assertEqual((r["state"], r["control_owner"], r["autonomy_enabled"]),
                         ("DRIVER_CONTROL", "driver", False))
        for _ in range(100):
            r = m.update("VIOLATION", True, 10.0, 0.4, 0.025, engage_request=True)
        self.assertEqual(r["state"], "DRIVER_CONTROL")

    def test_repeated_ack_is_one_takeover_event(self):
        m = self.machine_in_tor()
        for _ in range(50):
            r = m.update("VIOLATION", True, 10.0, 0.4, 0.025)
        self.assertEqual((r["tor_events"], r["takeover_events"]), (1, 1))

    def test_tor_is_counted_per_transition_not_per_frame(self):
        m = self.machine_in_tor()
        for _ in range(100):
            r = m.update("VIOLATION", False, 10.0, 0.4, 0.025)
        self.assertEqual(r["tor_events"], 1)

    def test_a_simulated_ack_is_never_reported_as_a_verified_human(self):
        m = self.machine_in_tor()
        self.assertFalse(m.update("VIOLATION", True, 10.0, 0.4, 0.025)["human_takeover_verified"])

    def test_a_repeated_frame_is_no_progress(self):
        m = self.machine_in_tor()
        for _ in range(1000):
            r = m.update("VIOLATION", False, 10., .4, 0.0)
        self.assertEqual(r["state"], "TAKEOVER_REQUEST")

    def test_invalid_dt_fails_toward_the_mrm(self):
        for bad in (NAN, -0.025, math.inf, None, True):
            with self.subTest(dt=bad):
                m = self.machine_in_tor()
                self.assertEqual(m.update("VIOLATION", False, 10., .4, bad)["state"],
                                 "MRM_EXECUTING")

    def test_invalid_speed_never_counts_as_stopped(self):
        m = L3StateMachine()
        m.update("VIOLATION", False, 10.0, 0.4, 0.025, critical=True)
        for bad in (NAN, -1.0):
            self.assertEqual(m.update("VIOLATION", False, bad, .4, .025)["state"],
                             "MRM_EXECUTING")

    def test_invalid_mu_gives_the_gentlest_mrm_not_nan(self):
        m = L3StateMachine()
        r = m.update("VIOLATION", False, 10.0, NAN, 0.025, critical=True)
        self.assertTrue(math.isfinite(r["target_decel_ms2"]))
        self.assertGreater(r["target_decel_ms2"], 0.0)



# ---------------------------------------------------------------------------
# WP06 - measured control rate, simulated time and real-time factor.
# ---------------------------------------------------------------------------

class ControlTimingTests(unittest.TestCase):
    def window(self, cycles, wall_span, sim_step=0.025, start=100.0):
        w = ControlWindow()
        w.start(start)
        for i in range(cycles):
            w.cycle(start + (i + 1) * wall_span / cycles, sim_time_s=i * sim_step, frame_id=i)
        return w

    def test_forty_cycles_in_two_seconds_is_twenty_hertz(self):
        w = self.window(40, 2.0)
        w.stop(102.0)
        summary = w.summary(40, 0.025)
        self.assertEqual(summary["status"], "MEASURED")
        self.assertAlmostEqual(summary["delivered_control_hz"], 20.0)
        self.assertAlmostEqual(summary["simulated_elapsed_s"], 39 * 0.025)
        self.assertAlmostEqual(summary["real_time_factor"], round(0.975 / 2.0, 4))

    def test_slow_teardown_after_stop_does_not_change_the_rate(self):
        w = self.window(40, 1.0)
        w.stop(101.0)
        w.stop(106.0)                       # a cleanup path calling stop again
        self.assertAlmostEqual(w.summary(40, 0.025)["delivered_control_hz"], 40.0)

    def test_duplicate_frames_are_not_new_updates(self):
        w = ControlWindow()
        w.start(0.0)
        for i, frame in enumerate((1, 2, 2, 3, 3, 3, 4)):
            w.cycle(0.025 * (i + 1), frame_id=frame)
        w.stop(0.175)
        summary = w.summary(40, 0.025)
        self.assertEqual((summary["unique_control_updates"], summary["duplicate_frames_skipped"]),
                         (4, 3))

    def test_too_few_cycles_is_not_evaluated_not_zero(self):
        w = self.window(1, 0.025)
        w.stop(100.025)
        summary = w.summary(40, 0.025)
        self.assertEqual(summary["status"], "NOT_EVALUATED")
        self.assertIsNone(summary["delivered_control_hz"])

    def test_interval_tail_and_deadline_misses_are_reported_with_counts(self):
        w = ControlWindow()
        w.start(0.0)
        t = 0.0
        for i, gap in enumerate([0.025] * 98 + [0.2]):
            t += gap
            w.cycle(t, frame_id=i)
        w.stop(t)
        summary = w.summary(40, 0.025)
        self.assertEqual(summary["control_interval_ms"]["count"], 98)
        self.assertAlmostEqual(summary["control_interval_ms"]["max"], 200.0)
        self.assertEqual(summary["deadline_misses"], 1)


class SimClockTests(unittest.TestCase):
    def test_skipped_world_frames_give_the_real_step(self):
        clock = SimClock(0.025)
        self.assertEqual(clock.step(2.500), (0.025, "first"))
        dt, status = clock.step(2.600)          # world frames 100 -> 104
        self.assertAlmostEqual(dt, 0.1)
        self.assertEqual(status, "ok")

    def test_repeated_frame_is_zero_not_a_step(self):
        clock = SimClock(0.025)
        clock.step(1.0)
        self.assertEqual(clock.step(1.0), (0.0, "ok"))

    def test_backwards_or_non_finite_time_is_invalid(self):
        clock = SimClock(0.025)
        clock.step(1.0)
        for bad in (0.5, NAN, None, math.inf):
            dt, status = clock.step(bad)
            self.assertEqual(status, "invalid")
            self.assertTrue(math.isnan(dt))

    def test_a_long_jump_is_flagged_as_a_gap(self):
        clock = SimClock(0.025, max_gap_s=0.25)
        clock.step(1.0)
        dt, status = clock.step(3.0)
        self.assertEqual((dt, status), (2.0, "gap"))
        self.assertEqual(clock.summary()["counts"]["gap"], 1)



# ---------------------------------------------------------------------------
# WP04/WP05 - case statuses, reaction latency v2, suite accounting.
# ---------------------------------------------------------------------------

def _case(name="HardBrake", seed=42, weather="clear", status="PASS", **extra):
    row = {"name": name, "seed": seed, "weather": weather, "fault": "none",
           "triggered": True, "collisions": 0, "status": status,
           "pass": status == "PASS"}
    row.update(extra)
    return row


class CaseStatusTests(unittest.TestCase):
    def assess(self, kpi=None, **kw):
        args = dict(triggered=True, reacted=True, reaction_delay_s=0.3,
                    braked=True, evaded=False)
        args.update(kw)
        return assess_scenario("probe", "crossing",
                               {"collisions": 0} if kpi is None else kpi, **args)

    def test_clean_case_is_pass(self):
        self.assertEqual(self.assess()["status"], "PASS")

    def test_unjudgeable_cases_are_invalid_not_fail(self):
        for kw, kpi in (({"triggered": False}, None),
                        ({}, {}),
                        ({"reaction_delay_s": NAN}, None),
                        ({"reaction_delay_s": -0.2}, None),
                        ({"scenario_errors": ["walker crossing command: RuntimeError"]}, None),
                        ({"sensor_frame_errors": True}, None)):
            with self.subTest(kw=kw, kpi=kpi):
                v = self.assess(kpi, **kw)
                self.assertEqual(v["status"], "INVALID")
                self.assertFalse(v["pass"])

    def test_an_observed_failure_is_never_hidden_behind_invalid(self):
        v = self.assess({"collisions": 2}, reaction_delay_s=NAN)
        self.assertEqual(v["status"], "FAIL")
        self.assertTrue(v["failure_reasons"])
        self.assertTrue(v["invalid_reasons"])

    def test_invalid_threshold_fails_fast(self):
        for bad in (NAN, 0.0, -1.0, None, True):
            with self.assertRaises(ValueError):
                self.assess(max_reaction_s=bad)

    def test_cleanup_failure_turns_pass_into_error_and_keeps_fail(self):
        row = finalize_after_cleanup(_case(), ["destroy owned actors: timeout"])
        self.assertEqual((row["status"], row["pass"]), ("ERROR", False))
        self.assertFalse(row["cleanup"]["verified"])
        row = finalize_after_cleanup(_case(status="FAIL"), ["x"])
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(finalize_after_cleanup(_case(), [])["cleanup"]["verified"])


class ReactionTimingTests(unittest.TestCase):
    def test_reaction_after_hazard(self):
        t = reaction_timing(100, 104, 104, 0.025)
        self.assertAlmostEqual(t["reaction_delay_s"], 0.1)
        self.assertFalse(t["preemptive_response"])

    def test_reaction_that_ended_before_the_hazard_is_not_a_response(self):
        """v1 clamped this to an instant 0 s response."""
        t = reaction_timing(100, 80, None, 0.025)
        self.assertFalse(t["reacted_to_hazard"])
        self.assertIsNone(t["reaction_delay_s"])
        self.assertAlmostEqual(t["first_reaction_minus_hazard_s"], -0.5)

    def test_preemptive_response_still_active_at_the_hazard(self):
        t = reaction_timing(100, 80, 100, 0.025)
        self.assertTrue(t["preemptive_response"])
        self.assertEqual(t["reaction_delay_s"], 0.0)

    def test_no_hazard_no_latency(self):
        self.assertIsNone(reaction_timing(None, 10, None, 0.025)["reaction_delay_s"])

    def test_a_pre_hazard_reaction_that_stopped_gets_the_later_latency(self):
        t = reaction_timing(100, 80, 130, 0.025)
        self.assertAlmostEqual(t["reaction_delay_s"], 0.75)
        self.assertFalse(t["preemptive_response"])


class SuiteAccountingTests(unittest.TestCase):
    WEATHERS = ["clear", "light_rain", "heavy_rain", "fog", "storm"]
    CORE = ["C1", "C2", "C3", "C4", "C5", "C6"]

    def weather_suite(self, rows):
        planned = [case_id({"name": n, "seed": 42, "weather": w, "fault": "none"})
                   for w in self.WEATHERS for n in self.CORE]
        meta = {"suite_policy": "all_planned_cases", "planned_case_ids": planned,
                "core_scenarios": self.CORE}
        return build_report(rows, meta=meta)

    def full_weather_rows(self):
        return [_case(n, weather=w) for w in self.WEATHERS for n in self.CORE]

    def test_thirty_case_weather_suite_is_judged_on_its_own_matrix(self):
        report = self.weather_suite(self.full_weather_rows())
        self.assertEqual(report["acceptance_gate"]["status"], "PASS")
        self.assertEqual(report["suite"]["planned"], 30)
        self.assertEqual(report["suite"]["not_run"], 0)

    def test_one_failure_fails_the_planned_suite(self):
        rows = self.full_weather_rows()
        rows[3] = dict(rows[3], status="FAIL", **{"pass": False})
        self.assertEqual(self.weather_suite(rows)["acceptance_gate"]["status"], "FAIL")

    def test_missing_cases_are_not_run_and_block_pass(self):
        report = self.weather_suite(self.full_weather_rows()[:20])
        self.assertEqual(report["suite"]["not_run"], 10)
        self.assertEqual(report["acceptance_gate"]["status"], "NOT_EVALUATED")

    def test_error_or_invalid_case_blocks_pass(self):
        for status in ("ERROR", "INVALID"):
            rows = self.full_weather_rows()
            rows[0] = dict(rows[0], status=status, **{"pass": False})
            with self.subTest(status=status):
                report = self.weather_suite(rows)
                self.assertEqual(report["acceptance_gate"]["status"], "INVALID")
                self.assertEqual(report["suite"][status.lower()], 1)

    def test_unplanned_case_makes_the_suite_invalid(self):
        rows = self.full_weather_rows() + [_case("Surprise")]
        self.assertEqual(self.weather_suite(rows)["acceptance_gate"]["status"], "INVALID")

    def test_core_fail_with_catalog_subgate_met_is_overall_fail(self):
        names = [f"S{i}" for i in range(15)]
        core = names[:6]
        rows = [_case(n, seed=sd) for sd in (42, 1337, 2026) for n in names]
        rows[0] = dict(rows[0], status="FAIL", **{"pass": False})       # a core case
        report = build_report(rows, meta={"core_scenarios": core})
        self.assertTrue(report["acceptance_gate"]["catalog"]["pass"])   # 44/45
        self.assertEqual(report["acceptance_gate"]["status"], "FAIL")

    def test_exit_codes(self):
        self.assertEqual(exit_code_for({"acceptance_gate": {"status": "PASS"}}), 0)
        self.assertEqual(exit_code_for({"acceptance_gate": {"status": "FAIL"}}), 1)
        self.assertEqual(exit_code_for({"acceptance_gate": {"status": "INVALID"}}), 2)
        self.assertEqual(exit_code_for({"acceptance_gate": {"status": "NOT_EVALUATED"}}), 3)
        self.assertEqual(exit_code_for({}), 3)

    def test_report_file_is_strict_json_and_written_atomically(self):
        import json
        import os
        import tempfile
        rows = [_case(min_ttc_s=math.inf, reaction_delay_s=NAN)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "r.json")
            write_report(path, rows, meta={"planned_case_ids": [case_id(rows[0])]})
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertFalse(os.path.exists(path + ".tmp"))
        report = json.loads(text)                     # strict: no NaN/Infinity tokens
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertIsNone(report["scenarios"][0]["reaction_delay_s"])



# ---------------------------------------------------------------------------
# WP05/WP08 - KPI summary and the report generator do not invent numbers.
# ---------------------------------------------------------------------------

class KpiEvidenceTests(unittest.TestCase):
    def test_zero_frames_is_not_evaluated(self):
        summary = KpiRecorder(0.025).summary()
        self.assertEqual(summary["status"], "NOT_EVALUATED")
        self.assertFalse(summary["success"])
        self.assertIsNone(summary["mean_speed_kmh"])

    def test_raw_extremes_are_kept_next_to_the_comfort_values(self):
        kpi = KpiRecorder(0.025)
        for speed in (10.0, 10.0, 9.9, 5.0, 5.0):     # 4.9 m/s in one tick = 196 m/s^2
            kpi.add(0.0, speed)
        summary = kpi.summary()
        self.assertEqual(summary["decel_samples_above_cap"], 1)
        self.assertGreater(summary["raw_max_decel_ms2"], summary["max_decel_ms2"])
        self.assertGreater(summary["jerk_samples_above_cap"], 0)
        self.assertIn("not matched to collision", summary["comfort_filter"]["basis"])


class ValidationReportGeneratorTests(unittest.TestCase):
    def test_missing_latency_is_na_not_zero(self):
        rows = [{"stage_metrics": {"stages": {}}}, {"stage_metrics": {}}]
        value, have, total_cases = gvr.worst(
            rows, lambda r: gvr.stage(r, "safety_loop", "p99_ms"))
        self.assertIsNone(value)
        self.assertEqual((have, total_cases), (0, 2))
        self.assertEqual(gvr.verdict(value, 25.0, False), "NOT EVALUATED")
        self.assertEqual(gvr.fmt(value, "ms", (have, total_cases)), "N/A (0/2 cases)")

    def test_worst_case_reports_its_coverage(self):
        rows = [{"x": 3.0}, {"x": None}, {"x": NAN}, {"x": 7.0}]
        self.assertEqual(gvr.worst(rows, lambda r: r["x"]), (7.0, 2, 4))

    def test_a_count_missing_in_any_case_is_not_summed_as_zero(self):
        self.assertIsNone(gvr.total([{"collisions": 0}, {}], "collisions"))
        self.assertIsNone(gvr.total([{"collisions": 0}, {"collisions": -1}], "collisions"))
        self.assertEqual(gvr.total([{"collisions": 0}, {"collisions": 2}], "collisions"), 2)

    def test_zero_is_not_a_passing_latency_bar(self):
        self.assertEqual(gvr.bar(None, 25.0, higher=False), "·" * 24)



class L3HarnessReportTests(unittest.TestCase):
    def profile(self, name, expect="NORMAL", actual="NORMAL"):
        return assess_profile(name, expect, actual, {"collisions": 0}, False, 0, 0, 0)

    def test_distinct_profiles_are_distinct_cases(self):
        """Profile rows have no "name"; they must not all collapse into one
        case and be reported as duplicates."""
        rows = [self.profile("clear"), self.profile("fog", "DEGRADED", "DEGRADED")]
        meta = {"suite_policy": "all_planned_cases",
                "planned_case_ids": [f"{n}|seed=None|weather=None|fault=None"
                                     for n in ("clear", "fog")]}
        report = build_report(rows, meta=meta)
        self.assertEqual(report["acceptance_gate"]["duplicate_case_rows"], 0)
        self.assertEqual(report["acceptance_gate"]["status"], "PASS")

    def test_undeclared_expected_odd_is_invalid_not_a_self_graded_pass(self):
        row = self.profile("custom", expect=None, actual="NORMAL")
        self.assertEqual(row["status"], "INVALID")
        self.assertFalse(row["pass"])


if __name__ == "__main__":
    unittest.main()
