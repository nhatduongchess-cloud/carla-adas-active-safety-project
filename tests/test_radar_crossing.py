"""Crossing prediction, non-conflicts, scan gaps and ego-motion regression tests."""

import gzip
import json
import math
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np

from modules.active_safety import ActiveSafetySystem
from modules.mot_tracker import MultiObjectTracker
from modules.radar_crossing import radar_reference_pose
from modules.scenario_acceptance import assess_scenario


class RadarCrossingTests(unittest.TestCase):
    def test_recorded_storm_radar_precedes_direct_corridor(self):
        path = Path(__file__).parent / "fixtures/safety/town02_storm_radar_prefix.json.gz"
        with gzip.open(path, "rt", encoding="utf8") as stream:
            case = json.load(stream)
        direct, predictive = ActiveSafetySystem(), ActiveSafetySystem()
        responses = []
        for row in case["frames"]:
            targets = [dict(distance_m=x, lateral_m=y, closing_speed_ms=v) for x, y, v in row["targets"]]
            args = (row["speed_m_s"], [], [], case["dt"])
            kwargs = dict(path_points=row["path"], radar_targets=targets)
            self.assertEqual(direct.update(*args, **kwargs).state, "NORMAL")
            decision = predictive.update(*args, **kwargs, radar_pose=row["pose"])
            if decision.action == "BRAKE":
                responses.append((row["frame"], decision.threat["source"]))
        self.assertEqual(responses, [(case["frame_of_first_sent_aeb"], "radar_crossing")])
        self.assertLessEqual((responses[0][0] - case["hazard_frame"]) * case["dt"], 1.0)

    @staticmethod
    def decisions(vx=0.0, vy=-1.6, side=3.6, yaw_rate=0.0, dropout=None):
        safety = ActiveSafetySystem()
        decisions = []
        for frame in range(41):
            t = frame * 0.025
            yaw = t * yaw_rate
            x = 18.0 + (vx - 8.0) * t
            y = side + vy * t
            c, s = math.cos(yaw), math.sin(yaw)
            # Same world-fixed path represented in each rotating sensor frame.
            path = [(0.0, 0.0), (40.0 * c, -40.0 * s)]
            # RadarProcessor emits NumPy scalars. SafetyInput's replay schema
            # requires native Python numbers even when radar wins the decision.
            targets = [{"distance_m": np.float64(x * c + y * s), "lateral_m": np.float64(-x * s + y * c)}]
            if dropout and dropout(frame):
                targets = []
            decisions.append(
                safety.update(
                    8.0, [], [], 0.025, path_points=path, radar_targets=targets, radar_pose=(8.0 * t, 0.0, yaw)
                )
            )
            if decisions[-1].threat.get("source") == "radar_crossing":
                assert type(safety.last_input.nearest_distance_m) is float
                assert type(safety.last_input.ttc_s) is float
        return decisions

    def test_crossing_each_side_has_brake_command_by_deadline(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                decisions = self.decisions(side=sign * 3.6, vy=-sign * 1.6)
                first = next(d for d in decisions if d.state != "NORMAL")
                self.assertEqual(first.threat["source"], "radar_crossing")
                self.assertTrue(any(d.action == "BRAKE" and d.brake == 1.0 for d in decisions))

    def test_nonconflicts_stay_normal(self):
        for case in (
            dict(vy=0.0),
            dict(vx=1.5, vy=0.0),
            dict(vy=1.6),
            dict(vy=0.0, yaw_rate=0.25),
            dict(vy=-1.0, side=5.0),
        ):
            with self.subTest(case=case):
                self.assertTrue(all(d.state == "NORMAL" for d in self.decisions(**case)))

    def test_isolated_hits_do_not_form_a_crossing_history(self):
        self.assertTrue(all(d.state == "NORMAL" for d in self.decisions(dropout=lambda frame: frame % 9 != 0)))

    def test_sensor_mount_offset_rotates_with_vehicle(self):
        transform = SimpleNamespace(location=SimpleNamespace(x=3.0, y=4.0), rotation=SimpleNamespace(yaw=90.0))
        x, y, yaw = radar_reference_pose(transform, -0.5, -0.1)
        self.assertAlmostEqual(x, 3.1)
        self.assertAlmostEqual(y, 3.5)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_person_track_rejects_off_bearing_radar_return(self):
        for lateral, expected in ((2.5, False), (4.1, True)):
            with self.subTest(lateral=lateral):
                mot = MultiObjectTracker()
                mot.update(
                    [{"class": "Person", "distance_m": 9.0, "lateral_m": 4.0}],
                    radar_measurements=[{"distance_m": 9.5, "lateral_m": lateral}],
                )
                self.assertEqual("radar" in mot.tracks[0].sources, expected)

    def test_person_track_requires_conflict_at_arrival_and_resolved_motion(self):
        def decision(x, y, vx, vy, variance):
            person = SimpleNamespace(
                class_name="Person",
                pos=(x, y),
                vel=(vx, vy),
                ego_compensated=True,
                velocity_covariance=np.eye(2) * variance,
            )
            return ActiveSafetySystem().update(8.15, [], [], 0.025, tracks=[person])

        # Standing-walker control, frame 35. The old tracker treated time to
        # the padded lane edge (0.75 s) as collision time.
        self.assertEqual(decision(11.182, 3.210, 0.304, -0.987, 0.01).state, "NORMAL")
        self.assertEqual(decision(12.0, 3.4, 0.0, -1.5, 4.0).state, "NORMAL")
        self.assertEqual(decision(12.0, 3.4, 0.0, -1.5, 0.01).action, "BRAKE")

    def test_sidewalk_control_requires_valid_exposure_and_no_brake(self):
        def verdict(brake=False, errors=None):
            return assess_scenario(
                "StandingPedestrian",
                "sidewalk",
                {"collisions": 0, "min_distance_m": 2.0},
                triggered=True,
                reacted=brake,
                reaction_delay_s=None,
                braked=brake,
                evaded=False,
                scenario_errors=errors,
            )["status"]

        self.assertEqual(verdict(), "PASS")
        self.assertEqual(verdict(brake=True), "FAIL")
        self.assertEqual(verdict(errors=["walker unavailable"]), "INVALID")
