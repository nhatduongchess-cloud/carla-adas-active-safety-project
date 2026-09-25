"""Regression tests for the 2026-09-25 evidence review.

That review arrived as a script: fifteen offline probes, each reproducing one
suspected defect against the code as it stood. This file turns every probe that
exposed a real defect into an assertion of the corrected behaviour, so none of
them can come back quietly.

Every test here runs without CARLA. The one piece of arbitration that lives in
a CARLA-importing module was moved into `modules/control_arbitration.py` so it
could be tested here rather than only on a machine with the simulator client.

Probes that did NOT expose a defect are recorded at the bottom, with the reason,
so the next reader does not have to re-derive why they were left alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from modules.clearance import clearance, polygon_gap
from modules.control_arbitration import longitudinal_override
from modules.fault_acceptance import assess_fault_behavior
from modules.kpi import KpiRecorder
from modules.l3_report import assess_profile, build_report
from modules.mrm_controller import L3StateMachine
from modules.odd_monitor import ODDMonitor
from modules.scenario_acceptance import assess_scenario
from modules.scenario_library import RunningScenario
from modules.sensor_health import SensorHealthMonitor

ROOT = Path(__file__).resolve().parents[1]
NAN = math.nan


# ---------------------------------------------------------------------------
# 1. An active MRM must not veto a harder AEB brake.
# ---------------------------------------------------------------------------

class MrmAebArbitrationTests(unittest.TestCase):
    MRM = {"override": True, "state": "MRM_EXECUTING", "target_decel_ms2": 1.0, "hazard": True}

    def test_aeb_full_brake_survives_an_active_mrm(self):
        """The probe: AEB asked for 1.0, the vehicle got 0.167."""
        cmd = longitudinal_override(self.MRM, NS(action="BRAKE", brake=1.0))
        self.assertEqual(cmd["brake"], 1.0)
        self.assertEqual(cmd["source"], "aeb")

    def test_mrm_sets_the_pedal_when_it_is_the_stronger_request(self):
        mrm = dict(self.MRM, target_decel_ms2=4.5)
        cmd = longitudinal_override(mrm, NS(action="BRAKE", brake=0.3))
        self.assertAlmostEqual(cmd["brake"], 0.75)
        self.assertEqual(cmd["source"], "mrm")

    def test_mrm_alone_keeps_its_comfort_deceleration(self):
        cmd = longitudinal_override(self.MRM, NS(action="DRIVE", brake=0.0))
        self.assertAlmostEqual(cmd["brake"], 1.0 / 6.0)

    def test_safe_stop_holds_full_brake_and_hand_brake(self):
        cmd = longitudinal_override({"override": True, "state": "SAFE_STOP"}, NS(action="DRIVE"))
        self.assertEqual((cmd["brake"], cmd["hand_brake"]), (1.0, True))

    def test_an_invalid_brake_request_becomes_full_braking_not_none(self):
        cmd = longitudinal_override(dict(self.MRM, target_decel_ms2=NAN), NS(action="DRIVE"))
        self.assertEqual(cmd["brake"], 1.0)
        cmd = longitudinal_override(self.MRM, NS(action="BRAKE", brake=NAN))
        self.assertEqual(cmd["brake"], 1.0)

    def test_no_override_means_no_l3_command(self):
        self.assertIsNone(longitudinal_override({"override": False}, NS(action="BRAKE", brake=1.0)))


# ---------------------------------------------------------------------------
# 2. An ODD that cannot be measured is not an ODD that is satisfied.
# ---------------------------------------------------------------------------

class OddFailSafeTests(unittest.TestCase):
    def test_all_nan_inputs_are_a_violation_not_normal(self):
        r = ODDMonitor().classify({"visibility_m": NAN, "mu": NAN, "snr": NAN})
        self.assertEqual(r["state"], "VIOLATION")
        self.assertEqual(r["reason"], "odd_unmeasurable:visibility_m,mu,snr")

    def test_unmeasurable_asks_the_driver_first_rather_than_forcing_an_mrm(self):
        r = ODDMonitor().classify({"visibility_m": NAN, "mu": 0.9, "snr": 1.0})
        self.assertEqual(r["state"], "VIOLATION")
        self.assertFalse(r["critical"])

    def test_each_input_on_its_own_is_enough(self):
        good = {"visibility_m": 200.0, "mu": 0.9, "snr": 1.0}
        for field in good:
            with self.subTest(field=field):
                bad = dict(good, **{field: NAN})
                self.assertEqual(ODDMonitor().classify(bad)["state"], "VIOLATION")

    def test_finite_clear_conditions_are_still_normal(self):
        r = ODDMonitor().classify({"visibility_m": 200.0, "mu": 0.9, "snr": 1.0})
        self.assertEqual((r["state"], r["reason"]), ("NORMAL", None))


# ---------------------------------------------------------------------------
# 3. Losing every enabled range sensor is losing range sensing.
# ---------------------------------------------------------------------------

def _health(enabled, lidar_ok, radar_ok=True, frames=5):
    h = SensorHealthMonitor(enabled, 3)
    for i in range(frames):
        h.next_frame()
        h.observe("lidar", i, i * 0.025, lidar_ok)
        if enabled.get("radar"):
            h.observe("radar", i, i * 0.025, radar_ok)
    return h


class RangeSensingTests(unittest.TestCase):
    RADAR_OFF = {"camera": True, "lidar": True, "radar": False}
    BOTH = {"camera": True, "lidar": True, "radar": True}

    def test_lidar_lost_with_radar_disabled_is_total_range_loss(self):
        """The probe: 5 LiDAR misses, radar off, reported healthy."""
        h = _health(self.RADAR_OFF, lidar_ok=False)
        self.assertTrue(h.range_redundancy_lost)
        self.assertEqual(ODDMonitor().classify(
            {"visibility_m": 200.0, "mu": 0.9, "snr": 1.0}, h.summary())["state"], "VIOLATION")

    def test_radar_disabled_alone_is_not_a_loss(self):
        self.assertFalse(_health(self.RADAR_OFF, lidar_ok=True).range_redundancy_lost)

    def test_one_of_two_range_sensors_lost_keeps_range_sensing(self):
        self.assertFalse(_health(self.BOTH, lidar_ok=False, radar_ok=True).range_redundancy_lost)
        self.assertFalse(_health(self.BOTH, lidar_ok=True, radar_ok=False).range_redundancy_lost)

    def test_both_lost_is_still_a_loss(self):
        self.assertTrue(_health(self.BOTH, lidar_ok=False, radar_ok=False).range_redundancy_lost)

    def test_no_range_sensor_enabled_has_no_range_sensing(self):
        self.assertTrue(SensorHealthMonitor({"camera": True, "lidar": False, "radar": False}, 3)
                        .range_redundancy_lost)

    def test_a_disabled_sensor_has_no_availability_rather_than_perfect_availability(self):
        s = _health(self.RADAR_OFF, lidar_ok=True).summary()
        self.assertIsNone(s["availability"]["radar"])
        self.assertIs(s["enabled"]["radar"], False)
        self.assertEqual(s["range_sensors_enabled"], ["lidar"])


class FaultAcceptanceTests(unittest.TestCase):
    def _summary(self, enabled, availability, ever_lost):
        return {"enabled": enabled, "availability": availability,
                "range_redundancy_ever_lost": ever_lost}

    def test_lidar_fault_with_radar_disabled_must_escalate(self):
        s = self._summary({"lidar": True, "radar": False},
                          {"lidar": 0.2, "radar": None}, ever_lost=True)
        v = assess_fault_behavior("lidar-loss", s, odd_violation_seen=True, mrm_seen=True)
        self.assertTrue(v["pass"], v["reasons"])
        v = assess_fault_behavior("lidar-loss", s, odd_violation_seen=False, mrm_seen=False)
        self.assertFalse(v["pass"])

    def test_lidar_fault_with_radar_enabled_still_must_not_lose_range(self):
        s = self._summary({"lidar": True, "radar": True},
                          {"lidar": 0.2, "radar": 1.0}, ever_lost=True)
        v = assess_fault_behavior("lidar-loss", s)
        self.assertIn("single-sensor fault incorrectly lost range redundancy", v["reasons"])

    def test_a_fault_injected_into_a_disabled_sensor_tests_nothing(self):
        s = self._summary({"lidar": True, "radar": False},
                          {"lidar": 1.0, "radar": None}, ever_lost=False)
        v = assess_fault_behavior("radar-loss", s)
        self.assertIn("fault radar-loss targets radar, which is disabled", v["reasons"])


# ---------------------------------------------------------------------------
# 4. A takeover hands control to the driver and keeps it there.
# ---------------------------------------------------------------------------

class TakeoverTests(unittest.TestCase):
    def test_takeover_during_violation_does_not_oscillate(self):
        """The probe: TOR, L3_ACTIVE, TOR - automation 'active' in a violated ODD."""
        m = L3StateMachine()
        states = [m.update("VIOLATION", t, 10., .4, .025)["state"] for t in (False, True, True)]
        self.assertEqual(states, ["TAKEOVER_REQUEST", "DRIVER_CONTROL", "DRIVER_CONTROL"])

    def test_a_one_shot_takeover_never_leads_to_an_mrm(self):
        """Before: the re-issued TOR timed out and started an MRM 10.03 s later,
        with the driver already driving."""
        m = L3StateMachine()
        m.update("VIOLATION", False, 10., .4, .025)
        m.update("VIOLATION", True, 10., .4, .025)
        states = {m.update("VIOLATION", False, 10., .4, .025)["state"] for _ in range(800)}
        self.assertEqual(states, {"DRIVER_CONTROL"})

    def test_driver_control_neither_overrides_nor_flashes_hazards(self):
        m = L3StateMachine()
        m.update("VIOLATION", False, 10., .4, .025)
        r = m.update("VIOLATION", True, 10., .4, .025)
        self.assertFalse(r["override"])
        self.assertFalse(r["hazard"])

    def test_automation_returns_only_once_the_odd_is_normal(self):
        m = L3StateMachine()
        m.update("VIOLATION", False, 10., .4, .025)
        m.update("VIOLATION", True, 10., .4, .025)
        self.assertEqual(m.update("DEGRADED", False, 10., .5, .025)["state"], "DRIVER_CONTROL")
        self.assertEqual(m.update("NORMAL", False, 10., .9, .025)["state"], "L3_ACTIVE")


# ---------------------------------------------------------------------------
# 5. Acceptance fails closed.
# ---------------------------------------------------------------------------

def _scenario(kpi, delay):
    return assess_scenario("probe", "crossing", kpi, triggered=True, reacted=True,
                           reaction_delay_s=delay, braked=True, evaded=False)


class ScenarioAcceptanceTests(unittest.TestCase):
    OK = {"collisions": 0, "min_distance_m": 1.0}

    def test_nan_delay_and_nan_clearance_fail(self):
        v = _scenario({"collisions": 0, "min_distance_m": NAN}, NAN)
        self.assertFalse(v["pass"])
        self.assertEqual(len(v["reasons"]), 2)

    def test_negative_delay_is_a_timestamp_fault(self):
        v = _scenario(self.OK, -1.0)
        self.assertFalse(v["pass"])
        self.assertIn("timestamp fault", v["reasons"][0])

    def test_missing_collision_count_is_not_zero_collisions(self):
        v = _scenario({}, 0.1)
        self.assertEqual(v["reasons"], ["collision count was not observed"])

    def test_invalid_collision_counts_fail(self):
        # NaN, negative, fractional, a bool and a non-numeric string. A numeric
        # string such as "0" is still a valid count, as it always was.
        for bad in (NAN, -1, 1.5, True, "none"):
            with self.subTest(bad=bad):
                self.assertFalse(_scenario({"collisions": bad, "min_distance_m": 1.0}, 0.1)["pass"])

    def test_unmeasured_clearance_stays_allowed_as_documented(self):
        self.assertTrue(_scenario({"collisions": 0, "min_distance_m": None}, 0.1)["pass"])

    def test_a_clean_run_still_passes(self):
        self.assertTrue(_scenario(self.OK, 0.3)["pass"])


class L3ProfileTests(unittest.TestCase):
    def test_late_braking_is_now_a_gate(self):
        r = assess_profile("clear", "NORMAL", "NORMAL", {"collisions": 0}, False, 0, 0, 5)
        self.assertFalse(r["pass"])

    def test_missing_collision_count_fails(self):
        r = assess_profile("clear", "NORMAL", "NORMAL", {}, False, 0, 0, 0)
        self.assertEqual(r["reasons"], ["collision count was not observed"])

    def test_clean_profile_still_passes(self):
        r = assess_profile("clear", "NORMAL", "NORMAL", {"collisions": 0}, False, 0, 0, 0)
        self.assertTrue(r["pass"])


def _case(name, seed, ok=True, collisions=0, weather="clear"):
    return {"name": name, "seed": seed, "weather": weather, "fault": "none",
            "triggered": True, "collisions": collisions, "pass": ok}


class AcceptanceGateTests(unittest.TestCase):
    CORE = ["A", "B", "C", "D", "E", "F"]

    def test_duplicate_rows_make_the_gate_invalid(self):
        """The probe: 45 copies of one case reported PASS."""
        g = build_report([_case("one_case", 42)] * 45,
                         meta={"core_scenarios": ["one_case"]})["acceptance_gate"]
        self.assertEqual(g["status"], "INVALID")
        self.assertEqual(g["duplicate_case_rows"], 44)

    def test_one_core_scenario_at_eighteen_seeds_is_not_the_core_suite(self):
        rows = [_case("A", s) for s in range(18)] + [_case(f"x{i}", 0) for i in range(27)]
        g = build_report(rows, meta={"core_scenarios": self.CORE})["acceptance_gate"]
        self.assertEqual(g["core"]["missing_core_scenarios"], ["B", "C", "D", "E", "F"])
        self.assertNotEqual(g["status"], "PASS")

    def test_a_complete_distinct_suite_can_still_pass(self):
        rows = [_case(n, s) for n in self.CORE for s in (1, 2, 3)]
        rows += [_case(f"x{i}", s) for i in range(9) for s in (1, 2, 3)]
        g = build_report(rows, meta={"core_scenarios": self.CORE})["acceptance_gate"]
        self.assertEqual(g["status"], "PASS")

    def test_an_empty_suite_asserts_nothing_about_collisions(self):
        g = build_report([])["acceptance_gate"]
        self.assertIsNone(g["no_collision_in_valid_runs"])
        self.assertEqual(g["status"], "NOT_EVALUATED")

    def test_a_row_with_no_collision_count_is_not_collision_free(self):
        rows = [_case("A", 1)]
        del rows[0]["collisions"]
        self.assertIs(build_report(rows)["acceptance_gate"]["no_collision_in_valid_runs"], False)

    def test_published_reports_keep_their_verdicts(self):
        """The fix changes manufactured cases only. Every published report must
        get the same gate status from the new code as it recorded."""
        for name in ("head_catalog_3seed", "head_core_5weather",
                     "head_core_seed42", "head_doc_probe"):
            with self.subTest(report=name):
                r = json.loads((ROOT / "docs/benchmarks" / f"{name}.json").read_text(encoding="utf-8"))
                g = build_report(r["scenarios"], meta=r["run"])["acceptance_gate"]
                self.assertEqual(g["status"], r["acceptance_gate"]["status"])

    def test_default_title_names_no_manufacturer(self):
        self.assertNotIn("Mercedes", build_report([])["title"])


class KpiTests(unittest.TestCase):
    def test_a_run_with_no_frames_did_not_succeed(self):
        self.assertIs(KpiRecorder(0.025).summary()["success"], False)


# ---------------------------------------------------------------------------
# 6. Clearance is surface to surface; trigger distance is not moved.
# ---------------------------------------------------------------------------

def _car(x, y, yaw=0.0, half=(2.4, 1.0)):
    return NS(is_alive=True,
              get_location=lambda: NS(x=x, y=y),
              get_transform=lambda: NS(location=NS(x=x, y=y), rotation=NS(yaw=yaw)),
              bounding_box=NS(extent=NS(x=half[0], y=half[1]), location=NS(x=0., y=0.)))


class ClearanceTests(unittest.TestCase):
    def test_touching_cars_have_zero_clearance_not_centre_distance(self):
        gap, basis = clearance(_car(0, 0), _car(4.8, 0))
        self.assertEqual((round(gap, 6), basis), (0.0, "oriented_boxes"))

    def test_nose_to_tail_side_by_side_and_perpendicular(self):
        self.assertAlmostEqual(clearance(_car(0, 0), _car(5.8, 0))[0], 1.0)
        self.assertAlmostEqual(clearance(_car(0, 0), _car(0, 3.0))[0], 1.0)
        self.assertAlmostEqual(clearance(_car(0, 0), _car(3.9, 0, yaw=90))[0], 0.5)

    def test_overlapping_boxes_are_zero(self):
        square = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertEqual(polygon_gap(square, [(1, 1), (3, 1), (3, 3), (1, 3)]), 0.0)

    def test_a_box_without_orientation_falls_back_conservatively(self):
        """The probe's geometry: centres 3.0 m apart, 2 m half-length."""
        ego = NS(get_location=lambda: NS(x=0., y=0.))
        actor = NS(get_location=lambda: NS(x=3., y=0.), bounding_box=NS(extent=NS(x=2., y=1.)))
        self.assertEqual(clearance(ego, actor), (1.0, "radius"))

    def test_no_boxes_at_all_is_flagged_as_centre_distance(self):
        a = NS(get_location=lambda: NS(x=0., y=0.))
        b = NS(get_location=lambda: NS(x=3., y=4.))
        self.assertEqual(clearance(a, b), (5.0, "center"))

    def test_scenario_triggers_on_centre_distance_and_scores_on_clearance(self):
        """Trigger timing must not move; only the acceptance number changes."""
        ego, other = _car(0, 0), _car(4.8, 0)
        sc = RunningScenario([other], trigger_distance=4.9)
        sc.tick(1, ego, None)
        self.assertTrue(sc.triggered)
        self.assertAlmostEqual(sc.min_actor_distance_m, 4.8)
        self.assertAlmostEqual(sc.min_clearance_m, 0.0)
        self.assertEqual(sc.clearance_basis, "oriented_boxes")


# ---------------------------------------------------------------------------
# Probes that did not expose a defect, and why.
# ---------------------------------------------------------------------------
#
# weather_model_friction_at_extremes -> [0.9, 0.4]
#     Correct as physics: dry asphalt ~0.9, wet ~0.4, and CARLA models no ice or
#     standing water. What it does reveal is that the ODD monitor's mu < 0.3
#     VIOLATION and mu < 0.2 critical branches cannot be reached from weather in
#     this simulation. That is recorded in odd_monitor.py and ARCHITECTURE §12;
#     lowering the friction floor to reach a threshold would be tuning the data
#     to hit a gate.
#
# brake_latch_then_missing_geometry -> BRAKE x5, then DRIVE
#     Real and confirmed, but NOT fixed here. After a critical brake, five frames
#     with an empty corridor release the latch, and during the hold the brake
#     drops from 1.0 to 0.7. An empty corridor cannot be told apart from an
#     obstacle that fell into the LiDAR's near-field blind zone. Any fix changes
#     live AEB behaviour and has to be re-validated on the simulator before it
#     ships; it is tracked as an open gap in ARCHITECTURE §12.
#
# empty_suite_gate -> NOT_EVALUATED
#     The status was already right; only the vacuous "no collision" claim was
#     not, and that is covered above.
