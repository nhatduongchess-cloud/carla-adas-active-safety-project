"""Offline A03 endpoint regressions; no CARLA/model/process dependencies."""
import math
from types import SimpleNamespace
import unittest

from modules.active_safety import ActiveSafetySystem
from modules import road_geometry as geometry


def radar(x, closing=10., y=0.):
    return {'distance_m': x, 'lateral_m': y, 'closing_speed_ms': closing}


def obstacle(x, y=0.):
    return {'centroid': (x, y, 0.), 'point_count': 20}


def track(x, y=0., vx=-10., vy=0.):
    return SimpleNamespace(pos=(x, y), vel=(vx, vy), class_name='Car', ego_compensated=False)


class SafetyEndpointTests(unittest.TestCase):
    path = [(0., 0.), (38.5, 0.)]

    def test_finite_projection_contract_is_unchanged(self):
        self.assertEqual(geometry.project_to_path(70., 0., self.path), (38.5, 0., 31.5))

    def test_safety_projection_retains_endpoint_overflow(self):
        self.assertEqual(geometry.project_to_safety_path(70., 0., self.path), (70., 0., 31.5))
        for actual, expected in zip(geometry.project_to_safety_path(-5., 1., self.path),
                                    (-5., 1., math.hypot(5, 1))):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_audit_radar70_does_not_become38_5(self):
        d = ActiveSafetySystem(enable_evasion=False).update(
            10., [], [], .025, path_points=self.path, radar_targets=[radar(70., 30.)])
        self.assertEqual(d.threat['radar_distance_m'], 70.)
        self.assertEqual(d.threat['ttc_s'], round(70/30, 2))
        self.assertEqual(d.action, 'SLOW')  # warning remains; no forced DRIVE

    def test_far_radar_with_genuinely_critical_ttc_still_brakes(self):
        d = ActiveSafetySystem(enable_evasion=False).update(
            10., [], [], .025, path_points=self.path, radar_targets=[radar(70., 50.)])
        self.assertEqual(d.action, 'BRAKE')
        self.assertEqual(d.threat['ttc_s'], 1.4)

    def test_near10_15_20_all_geometry_sources_still_brake(self):
        for distance in (10., 15., 20.):
            for source in ('radar', 'lidar', 'tracker'):
                with self.subTest(distance=distance, source=source):
                    kwargs = dict(path_points=self.path)
                    lidar = [obstacle(distance)] if source == 'lidar' else []
                    if source == 'radar': kwargs['radar_targets'] = [radar(distance, 15.)]
                    if source == 'tracker': kwargs['tracks'] = [track(distance, vx=-15.)]
                    safety = ActiveSafetySystem(enable_evasion=False)
                    if source == 'lidar':
                        # LiDAR closing speed needs two measured positions, not
                        # a synthetic forced velocity or an uninitialized history.
                        safety.update(15., [], [obstacle(distance + 15. * .025)], .025, **kwargs)
                    d = safety.update(15., [], lidar, .025, **kwargs)
                    self.assertEqual(d.action, 'BRAKE')

    def test_lidar_and_tracker_far_ranges_are_not_clipped(self):
        s = ActiveSafetySystem()
        self.assertEqual(s._scan_corridors([obstacle(70.)], self.path)[0], 70.)
        distance, ttc = s._predictive([track(70., vx=-30.)], self.path)
        self.assertEqual(distance, 70.)
        self.assertAlmostEqual(ttc, 70/30)

    def test_endpoint_crossing_has_no_artificial_lidar_plateau(self):
        s = ActiveSafetySystem()
        measured = [s._scan_corridors([obstacle(x)], self.path)[0] for x in (39., 38.5, 38.)]
        for actual, expected in zip(measured, (39., 38.5, 38.)):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_offside_and_behind_targets_not_front_hazards(self):
        s = ActiveSafetySystem()
        for target in (radar(70., 30., 5.), radar(-5., 30., 0.)):
            self.assertTrue(math.isinf(s._radar_threat([target], self.path)[0]))
        nearest, _, gaps = s._scan_corridors([obstacle(-5., 3.)], self.path)
        self.assertTrue(math.isinf(nearest))
        self.assertEqual(gaps['right']['rear'], 5.)

    def test_shifted_path_start_must_not_hide_near_front_obstacle(self):
        path = [(5., 0.), (10., 0.)]
        s = ActiveSafetySystem()
        d=s.update(10.,[],[obstacle(2.)],.025,path_points=path)
        self.assertEqual(d.action,'BRAKE')
        self.assertEqual(d.threat['distance_m'],2.)

    def test_curve_interior_matches_original_projection(self):
        path = [(0., 0.), (10., 0.), (20., 5.), (30., 10.)]
        for point in ((16., 3.), (18., 0.), (10., -2.)):
            self.assertEqual(geometry.project_to_safety_path(*point, path),
                             geometry.project_to_path(*point, path))

    def test_curve_endpoint_preserves_terminal_tangent_distance(self):
        path = [(0., 0.), (10., 0.), (10., 10.)]
        along, lateral, residual = geometry.project_to_safety_path(10., 30., path)
        self.assertEqual((along, lateral, residual), (40., 0., 20.))

    def test_duplicate_endpoint_segments_do_not_lose_overflow(self):
        path = [(0.,0.), (0.,0.), (38.5,0.), (38.5,0.)]
        self.assertEqual(geometry.project_to_safety_path(70.,0.,path)[0],70.)
        self.assertAlmostEqual(geometry.project_to_safety_path(-5.,0.,path)[0],-5.,places=12)

    def test_missing_or_degenerate_path_uses_straight_fallback(self):
        for path in (None, [], [(0.,0.)], [(0.,0.), (0.,0.)]):
            with self.subTest(path=path):
                self.assertEqual(geometry.project_to_safety_path(70.,0.,path)[0],70.)
                safety=ActiveSafetySystem(enable_evasion=False)
                safety.update(15.,[],[obstacle(10.375)],.025,path_points=path)
                d=safety.update(
                    15.,[],[obstacle(10.)],.025,path_points=path)
                self.assertEqual(d.action,'BRAKE')

    def test_reversed_initial_tangent_does_not_hide_front_obstacle(self):
        path=[(0.,0.),(-40.,0.)]
        d=ActiveSafetySystem(enable_evasion=False).update(10.,[],[obstacle(2.)],.025,path_points=path)
        self.assertEqual(d.action,'BRAKE')
        self.assertEqual(d.threat['distance_m'],2.)

    def test_nonfinite_path_falls_back_without_hiding_valid_obstacle(self):
        for value in (float('nan'),float('inf')):
            path=[(0.,0.),(value,0.)]
            d=ActiveSafetySystem(enable_evasion=False).update(10.,[],[obstacle(2.)],.025,path_points=path)
            self.assertEqual(d.action,'BRAKE')
            self.assertEqual(d.threat['distance_m'],2.)

    def test_invalid_point_does_not_override_valid_near_geometry(self):
        for value in (float('nan'),float('inf')):
            result=geometry.project_to_safety_path(10.,value,self.path)
            self.assertTrue(all(math.isinf(v) for v in result))
            d=ActiveSafetySystem(enable_evasion=False).update(
                10.,[],[obstacle(2.)],.025,path_points=self.path,
                radar_targets=[radar(10.,30.,value)])
            self.assertEqual(d.action,'BRAKE')
            self.assertEqual(d.threat['source'],'lidar')

    def test_short_path_does_not_block_far_side_lane(self):
        path = [(0.,0.), (10.,0.)]
        s=ActiveSafetySystem()
        _, _, gaps=s._scan_corridors([obstacle(30.,3.)],path)
        self.assertEqual(gaps['right']['front'],30.)
        self.assertTrue(s._lane_change_clear('right',gaps,[track(30.,3.,vx=0.)],path,None))
        self.assertFalse(s._lane_change_clear('right',gaps,[track(-5.,3.,vx=0.)],path,None))

    def test_near_lidar_wins_over_corrected_far_radar(self):
        d=ActiveSafetySystem(enable_evasion=False).update(
            10.,[],[obstacle(4.)],.025,path_points=self.path,radar_targets=[radar(70.,30.)])
        self.assertEqual(d.action,'BRAKE')
        self.assertEqual(d.threat['source'],'lidar')
        self.assertEqual(d.threat['distance_m'],4.)

    def test_input_geometry_and_raw_measurements_are_not_mutated(self):
        import copy
        path=copy.deepcopy(self.path)
        targets=[radar(70.,30.)]
        before=copy.deepcopy((path,targets))
        ActiveSafetySystem().update(10.,[],[],.025,path_points=path,radar_targets=targets)
        self.assertEqual((path,targets),before)

    def test_predictive_cutin_distance_retains_overflow(self):
        distance, ttc=ActiveSafetySystem()._predictive(
            [track(70.,3.,vx=-1.,vy=-2.)],self.path)
        self.assertAlmostEqual(distance,69.25)
        self.assertAlmostEqual(ttc,.75)  # existing corridor-entry prediction policy


if __name__ == '__main__':
    unittest.main()
