"""B19 heading/footprint ambiguity; no simulator process required."""
import math
from types import SimpleNamespace as NS
import unittest

from modules import road_geometry as geometry
from modules.active_safety import ActiveSafetySystem


def waypoint(x=0., y=0., yaw=0., lane=1, width=4., kind='Driving'):
    return NS(transform=NS(location=NS(x=x, y=y, z=0.), rotation=NS(yaw=yaw)),
              lane_id=lane, road_id=2, section_id=0, lane_width=width,
              lane_type=kind, lane_change='NONE', is_junction=False,
              get_left_lane=lambda: None, get_right_lane=lambda: None,
              next=lambda distance: [])


def continue_lane(wp):
    def next_point(distance):
        angle = math.radians(wp.transform.rotation.yaw)
        point = wp.transform.location
        following = waypoint(point.x+distance*math.cos(angle),
                             point.y+distance*math.sin(angle),
                             wp.transform.rotation.yaw, wp.lane_id, wp.lane_width)
        continue_lane(following)
        return [following]
    wp.next = next_point


def scene(nearest, yaw=0., extent=(2.2, .9)):
    transform = NS(location=NS(x=0., y=0., z=.2), rotation=NS(yaw=yaw))
    ego = NS(get_transform=lambda: transform, get_location=lambda: transform.location,
             bounding_box=NS(extent=NS(x=extent[0], y=extent[1])))
    world = NS(get_map=lambda: NS(get_waypoint=lambda point: nearest))
    return world, ego


def ambiguous_scene():
    # Saved Town02 B18 pose expressed in the ego frame. CARLA yaw can exceed360.
    wrong = waypoint(.5442988628, -1.8407610523, -175.1566467285, lane=-1)
    correct = waypoint(.2065879316, 2.1449666626, 364.8433380127, lane=1)
    wrong.get_left_lane = lambda: correct
    correct.get_left_lane = lambda: wrong  # Opposing left links cycle in CARLA.
    continue_lane(wrong)
    continue_lane(correct)
    return (*scene(wrong), wrong, correct)


class HeadingWaypointTests(unittest.TestCase):
    def test_saved_pose_both_origins_follow_forward_reference(self):
        world, ego, wrong, correct = ambiguous_scene()
        for offset in (0., 1.5):
            with self.subTest(origin_offset=offset):
                path = geometry.build_ego_path(world, ego, origin_offset_x=offset)
                self.assertGreater(len(path), 2)
                self.assertTrue(all(x > 0 for x, y in path[1:]))
                self.assertGreater(path[1][1], 2.)

    def test_lane_context_uses_same_reference_without_granting_lane_change(self):
        world, ego, _, correct = ambiguous_scene()
        context = geometry.get_lane_context(world, ego)
        self.assertEqual(context['current_lane_id'], correct.lane_id)
        self.assertFalse(context['left_change_allowed'])
        self.assertFalse(context['right_change_allowed'])

    def test_aligned_nearest_keeps_existing_path_without_neighbor_queries(self):
        nearest = waypoint(y=.2, yaw=359.)
        continue_lane(nearest)
        def unexpected():
            raise AssertionError('normal path must not search adjacent lanes')
        nearest.get_left_lane = nearest.get_right_lane = unexpected
        world, ego = scene(nearest, yaw=-1.)
        path = geometry.build_ego_path(world, ego, lookahead_m=4.)
        self.assertEqual(len(path), 3)
        self.assertGreater(path[-1][0], 3.9)

    def test_shoulders_sidewalks_and_other_roads_are_not_route_candidates(self):
        for field, value in (('lane_type', 'Shoulder'), ('lane_type', 'Sidewalk'),
                             ('road_id', 9), ('section_id', 9)):
            with self.subTest(field=field, value=value):
                world, ego, _, candidate = ambiguous_scene()
                setattr(candidate, field, value)
                self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_does_not_cross_median_to_search_further_lanes(self):
        world, ego, wrong, correct = ambiguous_scene()
        median = waypoint(y=1., kind='Shoulder')
        median.get_left_lane = lambda: correct
        wrong.get_left_lane = lambda: median
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_wrong_way_center_does_not_snap_to_nonoverlapping_neighbor(self):
        world, ego, _, candidate = ambiguous_scene()
        candidate.transform.location.y = 4.0
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_missing_invalid_footprint_does_not_authorize_reference_switch(self):
        for invalid in (None, -1., float('nan'), float('inf')):
            with self.subTest(footprint=invalid):
                world, ego, _, _ = ambiguous_scene()
                if invalid is None:
                    del ego.bounding_box
                else:
                    ego.bounding_box.extent.y = invalid
                self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_no_aligned_neighbor_never_supplies_behind_only_control_path(self):
        nearest = waypoint(.5, -1.8, 185., lane=-1)
        continue_lane(nearest)
        world, ego = scene(nearest)
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_missing_map_waypoint_retains_fallback(self):
        world, ego = scene(None)
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_unresolved_junction_keeps_map_priority_context(self):
        nearest = waypoint(yaw=180.)
        nearest.is_junction = True
        world, ego = scene(nearest)
        context = geometry.get_lane_context(world, ego)
        self.assertTrue(context['is_junction'])
        self.assertIsNone(context['current_lane_id'])
        self.assertFalse(context['left_change_allowed'])
        self.assertFalse(context['right_change_allowed'])

    def test_right_neighbor_is_considered(self):
        world, ego, wrong, correct = ambiguous_scene()
        wrong.get_left_lane = lambda: None
        wrong.get_right_lane = lambda: correct
        self.assertGreater(geometry.build_ego_path(world, ego)[1][1], 2.)

    def test_offset_bounding_box_must_actually_overlap_candidate(self):
        world, ego, _, _ = ambiguous_scene()
        ego.bounding_box.location = NS(x=0., y=-2.)
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_rotated_bounding_box_uses_its_lateral_projection(self):
        world, ego, _, correct = ambiguous_scene()
        ego.bounding_box.extent.x = .4
        correct.transform.location.y = 2.9
        self.assertGreater(len(geometry.build_ego_path(world, ego)), 2)
        ego.bounding_box.rotation = NS(yaw=90.)
        self.assertEqual(geometry.build_ego_path(world, ego), geometry.straight_path())

    def test_near_forward_obstacle_remains_protected_with_corrected_path(self):
        world, ego, _, _ = ambiguous_scene()
        path = geometry.build_ego_path(world, ego, origin_offset_x=1.5)
        for index in (5, 8, 10):
            x, y = path[index]
            decision = ActiveSafetySystem(enable_evasion=False).update(
                15., [], [], .025, path_points=path,
                radar_targets=[{'distance_m': x, 'lateral_m': y, 'closing_speed_ms': 15.}])
            self.assertEqual(decision.action, 'BRAKE')

    def test_junction_intent_selection_is_unchanged(self):
        nearest = waypoint()
        branches = [waypoint(2., -1., -45.), waypoint(2., 0., 0.), waypoint(2., 1., 45.)]
        nearest.next = lambda distance: branches
        world, ego = scene(nearest)
        for intent, expected_y in (('left', -1.), (None, 0.), ('right', 1.)):
            self.assertEqual(geometry.build_ego_path(world, ego, lookahead_m=2.,
                                                    turn_intent=intent)[1][1], expected_y)


if __name__ == '__main__':
    unittest.main()
