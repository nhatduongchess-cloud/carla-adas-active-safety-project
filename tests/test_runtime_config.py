"""B18: radar capture follows world ticks without changing safety cadence."""
import json
import unittest
from unittest.mock import Mock, patch

import config as cfg
from modules.runtime_config import PERFORMANCE_PROFILES, configure_runtime
from modules.sensor_setup import configure_radar_blueprint


class RadarCadenceTests(unittest.TestCase):
    def setUp(self):
        original = {key: getattr(cfg, key) for key in cfg._PERFORMANCE_BASELINE}
        self.addCleanup(lambda: vars(cfg).update(original))

    def test_async_every_world_tick_in_all_profiles(self):
        for profile in PERFORMANCE_PROFILES:
            with self.subTest(profile=profile):
                stable_async, effective = configure_runtime(cfg, profile, "async-stable")
                self.assertTrue(stable_async)
                self.assertEqual(cfg.RADAR_SENSOR_TICK_S, 0.0)
                self.assertEqual(effective["radar_sensor_tick_s"], 0.0)
                self.assertEqual(effective["radar_cadence"], "every_world_tick")
                # Variable world time has no configured fixed radar Hz.
                self.assertIsNone(effective["radar_sensor_hz"])
                self.assertIsNone(json.loads(json.dumps(effective))["radar_sensor_hz"])
                self.assertEqual(cfg.CAMERA_SENSOR_TICK_S, 0.1)
                self.assertFalse(cfg.CAMERA_POSTPROCESS)
                self.assertEqual(cfg.LIDAR_SENSOR_TICK_S, cfg.FIXED_DELTA)
                self.assertEqual(effective["safety_control_hz"], cfg.FPS)

    def test_shared_blueprint_receives_every_tick_without_recalibration(self):
        configure_runtime(cfg, "low-memory", "async-stable")
        blueprint = Mock()
        blueprint.has_attribute.return_value = True
        self.assertIs(configure_radar_blueprint(blueprint, cfg), blueprint)
        attributes = dict(call.args for call in blueprint.set_attribute.call_args_list)
        self.assertEqual(attributes, {
            "sensor_tick": "0.0", "range": "80.0", "horizontal_fov": "35.0",
            "vertical_fov": "10.0", "points_per_second": "4000",
        })

    def test_synchronous_round_trip_restores_configured_world_cadence(self):
        configure_runtime(cfg, "low-memory", "async-stable")
        stable_async, effective = configure_runtime(cfg, "quality", "synchronous")
        self.assertFalse(stable_async)
        self.assertEqual(cfg.RADAR_SENSOR_TICK_S, 0.0)
        self.assertEqual(effective["radar_cadence"], "every_world_tick")
        self.assertEqual(effective["radar_sensor_hz"], cfg.FPS)
        self.assertEqual(cfg.LIDAR_SENSOR_TICK_S, cfg._PERFORMANCE_BASELINE["LIDAR_SENSOR_TICK_S"])
        self.assertEqual(cfg.CAM_WIDTH, cfg._PERFORMANCE_BASELINE["CAM_WIDTH"])

    def test_synchronous_explicit_interval_is_preserved(self):
        with patch.dict(cfg._PERFORMANCE_BASELINE, RADAR_SENSOR_TICK_S=0.05):
            _, effective = configure_runtime(cfg, "quality", "synchronous")
            self.assertEqual(cfg.RADAR_SENSOR_TICK_S, 0.05)
            self.assertEqual(effective["radar_sensor_tick_s"], 0.05)
            self.assertEqual(effective["radar_cadence"], "fixed_interval")
            self.assertEqual(effective["radar_sensor_hz"], 20.0)

    def test_profile_does_not_enable_disabled_radar_or_change_sign(self):
        sign = cfg.RADAR_VELOCITY_SIGN
        with patch.object(cfg, "ENABLE_RADAR", False):
            for mode in ("async-stable", "synchronous"):
                configure_runtime(cfg, "low-memory", mode)
                self.assertFalse(cfg.ENABLE_RADAR)
                self.assertEqual(cfg.RADAR_VELOCITY_SIGN, sign)
                self.assertTrue(cfg.RADAR_REQUIRE_SIGN_VALIDATION)


if __name__ == "__main__":
    unittest.main()
