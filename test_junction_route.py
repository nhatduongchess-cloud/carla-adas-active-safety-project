"""B19 route continuity and unresolved-map control boundaries."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import config as cfg
from modules import road_geometry as geometry
from modules.ego_control import EgoController
from modules.lane_source import map_lane_estimate
from test_road_waypoints import waypoint, continue_lane, scene


def junction_scene():
    incoming = waypoint(yaw=-35.)
    incoming.is_junction = True
    incoming.road_id = 390
    continue_lane(incoming)
    world, ego = scene(incoming, yaw=-35.)
    world.id = 1
    route = geometry.EgoRoute(world, ego)
    route.update(40.)
    wrong = waypoint(yaw=180.)
    wrong.is_junction = True
    wrong.road_id = 384
    world.get_map = lambda: NS(get_waypoint=lambda point: wrong)
    return world, ego, route, wrong


class JunctionRouteTests(unittest.TestCase):
    def test_recorded_390_to_384_transition_retains_curve_reference(self):
        # Exact pre-loss waypoint and successor from the read-only Town02 probe.
        current=waypoint(40.8505058,306.0369873,-21.3499947,lane=-1)
        following=waypoint(43.1694832,304.5774536,-43.0201454,lane=-1)
        for point in (current,following):
            point.road_id=390
            point.is_junction=True
        continue_lane(following)
        current.next=lambda distance:[following]
        world,ego=scene(current,yaw=-30.9726086,extent=(2.3958898,1.0817250))
        pose=ego.get_transform()
        pose.location.x,pose.location.y=40.2213135,304.9469299
        route=geometry.EgoRoute(world,ego); route.update(40.)
        wrong=waypoint(41.3219109,302.5349731,-180.039032)
        wrong.road_id=384; wrong.is_junction=True
        world.get_map=lambda:NS(get_waypoint=lambda point:wrong)
        pose.location.x,pose.location.y=41.3246078,303.9907837
        pose.rotation.yaw=-40.8228798
        route.update(40.)
        self.assertTrue(route.context['route_valid'])
        self.assertEqual(route.context['road_id'],390)
        self.assertGreater(len(route.path()),2)

    def test_opposing_crossing_waypoint_does_not_erase_selected_branch(self):
        world, ego, route, wrong = junction_scene()
        route.update(40.)
        self.assertEqual(route.context['road_id'], 390)
        self.assertTrue(route.context['route_valid'])
        self.assertTrue(route.context['is_junction'])
        self.assertGreater(len(route.path()), 2)

    def test_aligned_crossing_branch_cannot_steal_selected_route(self):
        _, _, route, wrong = junction_scene()
        wrong.transform.rotation.yaw = -70.
        continue_lane(wrong)
        route.update(40.)
        self.assertEqual(route.context['road_id'], 390)

    def test_safety_and_controller_use_one_frozen_world_reference(self):
        _, ego, route, _ = junction_scene()
        route.update(45.)
        control = route.path()
        safety = route.path(1.5)
        expected = [(x-1.5,y) for x,y in control[1:] if x>1.5]
        for a,b in zip(expected,safety[1:]):
            self.assertAlmostEqual(a[0],b[0])
            self.assertAlmostEqual(a[1],b[1])
        self.assertEqual(len(expected),len(safety)-1)
        ego.get_transform().location.x += 10.
        self.assertEqual(route.path(),control)  # Snapshot is frozen until update.

    def test_consumers_keep_their_original_40_and_45_m_horizons(self):
        _,_,route,_=junction_scene()
        route.update(45.)
        safety=route.path(1.5,40.)
        control=route.path(lookahead_m=45.)
        self.assertEqual(len(safety),21)
        self.assertEqual(len(control),24)
        self.assertAlmostEqual(safety[-1][0],control[20][0]-1.5)

    def test_teleport_cannot_reuse_old_junction_route(self):
        _, ego, route, _ = junction_scene()
        ego.get_transform().location.x = 100.
        route.update(40.)
        self.assertFalse(route.context['route_valid'])
        self.assertEqual(route.path(),[])

    def test_other_elevation_cannot_reuse_flat_route(self):
        _, ego, route, _ = junction_scene()
        ego.get_transform().location.z = 10.
        route.update(40.)
        self.assertFalse(route.context['route_valid'])

    def test_new_episode_discards_old_route(self):
        world, _, route, _ = junction_scene()
        world.id = 2
        route.update(40.)
        self.assertFalse(route.context['route_valid'])

    def test_wrong_heading_cannot_reuse_route(self):
        _, ego, route, _ = junction_scene()
        ego.get_transform().rotation.yaw = 80.
        route.update(40.)
        self.assertFalse(route.context['route_valid'])

    def test_unresolved_junction_is_invalid_but_retains_map_priority(self):
        nearest = waypoint(yaw=180.)
        nearest.is_junction = True
        world, ego = scene(nearest)
        route = geometry.EgoRoute(world,ego)
        route.update(40.)
        estimate = map_lane_estimate(1,1.,route.path())
        self.assertTrue(route.context['is_junction'])
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.confidence,0.)
        self.assertIsNotNone(estimate.reason)

    def test_dead_end_cannot_be_labelled_a_valid_straight_route(self):
        world, ego = scene(waypoint())
        route = geometry.EgoRoute(world,ego)
        route.update(40.)
        self.assertFalse(route.context['route_valid'])
        self.assertEqual(route.path(),[])

    def test_nonfinite_or_degenerate_map_geometry_is_not_confident(self):
        for points in ([],[(0.,0.)],[(0.,0.),(0.,0.)],[(0.,0.),(float('nan'),0.)]):
            with self.subTest(points=points):
                estimate=map_lane_estimate(1,1.,points)
                self.assertFalse(estimate.valid)
                self.assertEqual(estimate.confidence,0.)

    def test_valid_straight_map_remains_valid(self):
        estimate=map_lane_estimate(1,1.,[(0.,0.),(20.,0.)])
        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.confidence,1.)

    @patch('modules.ego_control.set_hazard_lights')
    def test_control_consumes_prepared_reference_without_second_map_query(self,_):
        nearest=waypoint()
        continue_lane(nearest)
        world,proxy=scene(nearest)
        ego=MagicMock()
        ego.get_transform=proxy.get_transform
        ego.bounding_box=proxy.bounding_box
        ego.get_velocity.return_value=NS(x=5.,y=0.,z=0.)
        controller=EgoController(ego,None,cfg,world=world)
        path,context=controller.path_context()
        self.assertTrue(context['route_valid'])
        self.assertIs(controller.route,controller.custom.route)
        def unexpected():
            raise AssertionError('controller must consume the safety reference')
        world.get_map=unexpected
        status=controller.apply(NS(action='DRIVE',state='NORMAL',threat={}),
                                {'override':False,'hazard':False},'NORMAL',1,
                                target_speed_kmh=20.,respect_traffic_controls=False)
        self.assertEqual(status['mode'],'custom')
        self.assertTrue(status['route_valid'])

    def test_junction_context_query_uses_prepared_route_for_lane_permission(self):
        from modules.ego_driving_stack import EgoDrivingStack
        world,ego,route,_=junction_scene()
        stack=EgoDrivingStack(world,ego,cfg,route=route)
        stack._route_prepared=True
        with patch('modules.ego_driving_stack.get_lane_context',side_effect=AssertionError('new nearest lookup')):
            self.assertFalse(stack.request_lane_change('left'))

    @patch('modules.ego_control.set_hazard_lights')
    def test_unresolved_route_uses_real_control_fault_stop_not_forced_aeb(self,_):
        nearest=waypoint(yaw=180.)
        nearest.is_junction=True
        world, proxy=scene(nearest)
        ego=MagicMock()
        ego.get_transform=proxy.get_transform
        ego.get_location=proxy.get_location
        ego.bounding_box=proxy.bounding_box
        ego.get_velocity.return_value=NS(x=0.,y=0.,z=0.)
        ego.is_at_traffic_light.return_value=False
        controller=EgoController(ego,None,cfg,world=world)
        path, context=controller.path_context()
        decision=NS(action='DRIVE',state='NORMAL',threat={})
        status=controller.apply(decision,{'override':False,'hazard':False},'NORMAL',1,
                                target_speed_kmh=20.,respect_traffic_controls=False)
        self.assertFalse(context['route_valid'])
        self.assertEqual(path,[])
        self.assertEqual(status['mode'],'custom_fault_safe_stop')
        self.assertEqual(ego.apply_control.call_args.args[0].brake,1.)
        self.assertEqual(ego.apply_control.call_args.args[0].throttle,0.)
        self.assertEqual(decision.action,'DRIVE')


if __name__ == '__main__':
    unittest.main()
