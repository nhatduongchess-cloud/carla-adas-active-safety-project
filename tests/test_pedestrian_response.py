import unittest
from types import SimpleNamespace as NS

from modules.active_safety import ActiveSafetySystem, SafetyDecision
from modules.lane_source import LaneSourceArbiter, map_lane_estimate
from modules.pedestrian_response import PedestrianResponse, walker_position
from modules.perception_contracts import LaneEstimate


class PedestrianResponseTests(unittest.TestCase):
    def test_crossing_clock_and_command_ownership(self):
        trace = PedestrianResponse()
        lane = NS(source="map", reason=None)
        person = {"class": "Person", "confidence": .87, "bbox": [0, 1, 2, 3]}
        follow = SafetyDecision("FOLLOW", "SLOW")
        brake = SafetyDecision("EMERGENCY_BRAKE", "BRAKE", brake=1.)
        applied = {"mode": "aeb_override", "command_sent": True, "applied": {"brake": 1.}}
        trace.observe(1, 51, None, .025, [person], [], 51, lane, brake, applied)
        self.assertTrue(all(value is None for value in trace.events.values()))

        track = NS(id=3, pos=(12., 2.), vel=(0., -1.), class_name="Person", ego_compensated=True)
        obs = trace.observe(5, 55, 4, .025, [person], [track], 53, lane, follow, {})
        self.assertEqual(obs["confirmed_person_tracks"][0]["id"], 3)
        self.assertEqual(trace.events["yolo_person"]["image_frame"], 53)
        self.assertEqual(trace.events["safety_response"]["delay_s"], .025)
        trace.observe(6, 56, 4, .025, [], [], None, lane, brake,
                      dict(applied, mode="l3_override", brake_source="mrm"))
        self.assertIsNone(trace.events["aeb_command_sent"])
        trace.observe(7, 57, 4, .025, [], [], None, lane, brake, applied)
        self.assertEqual(trace.events["aeb_request"]["delay_s"], .05)
        self.assertEqual(trace.events["aeb_command_sent"]["delay_s"], .075)

    def test_selected_corridor_changes_obstacle_membership(self):
        straight = map_lane_estimate(1, 1., [(0., 0.), (15., 0.)])
        curved = LaneEstimate(1, 1., "learned", confidence=.95, valid=True, offset_m=0.,
                              lane_width_m=3.5, centerline_m=[(0., 0.), (4., 2.6), (15., 5.)])
        arbiter = LaneSourceArbiter(stable_frames=1)
        def response(lane):
            return ActiveSafetySystem().update(
                8., [], [{"centroid": [4., 2.6, 0.], "point_count": 20}], .025,
                path_points=lane.centerline_m).action
        self.assertEqual(response(arbiter.select(curved, straight, now_s=1., max_age_ms=150)), "BRAKE")
        for fallback in (arbiter.select(curved, straight, now_s=1.2, max_age_ms=150),
                         arbiter.select(curved, straight, at_junction=True)):
            self.assertEqual(fallback.source, "map")
            self.assertEqual(response(fallback), "DRIVE")

    def test_walker_ground_truth_uses_ego_axes(self):
        ego = NS(location=NS(x=10., y=20.), rotation=NS(yaw=90.))
        actor = NS(id=99, get_location=lambda: NS(x=8., y=30.))
        position = walker_position(ego, actor)
        self.assertAlmostEqual(position["forward_m"], 10.)
        self.assertAlmostEqual(position["right_m"], 2.)
        self.assertIsNone(walker_position(ego, None))
