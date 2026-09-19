"""Offline acquisition checks: diagnostics must not change sample selection."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules"))
from sensor_runtime import SensorRig


class RadarReceiveTimestampTests(unittest.TestCase):
    def rig(self, asynchronous=True, enabled=True):
        return SensorRig(SimpleNamespace(tick=lambda: 100), None,
                         SimpleNamespace(ENABLE_RADAR=enabled,
                                         RADAR_OPTIONAL_TIMEOUT_S=0.001),
                         asynchronous)

    def read(self, rig, frame=100):
        rig._image_queue.put(SimpleNamespace(frame=frame, timestamp=2.5))
        rig._lidar_queue.put(SimpleNamespace(frame=frame, timestamp=2.5))
        with patch("sensor_runtime.decode_rgb", return_value="rgb"), \
                patch("sensor_runtime.decode_lidar", return_value="lidar"):
            return rig.read(500.)

    def test_async_radar_keeps_original_object_frame_and_receive_clock(self):
        rig = self.rig()
        radar = SimpleNamespace(frame=99, timestamp=2.475, raw_data=b"original")
        with patch("sensor_runtime.time.monotonic", return_value=499.95):
            rig._queue_radar(radar)
        sample = self.read(rig)
        self.assertIs(sample.radar_measurement, radar)
        self.assertEqual(sample.radar_received_monotonic_s, 499.95)
        self.assertEqual(sample.frame_id, 100)
        self.assertEqual(sample.lidar_timestamp, 2.5)
        self.assertEqual(sample.capture_timestamp, 500.)

    def test_sync_exact_radar_and_bounded_queue_preserve_selection(self):
        rig = self.rig(asynchronous=False)
        for frame in range(95, 101):
            with patch("sensor_runtime.time.monotonic", return_value=float(frame)):
                rig._queue_radar(SimpleNamespace(frame=frame, timestamp=frame/40.))
        self.assertEqual(rig._radar_queue.qsize(), 4)
        sample = self.read(rig)
        self.assertEqual(sample.radar_measurement.frame, 100)
        self.assertEqual(sample.radar_received_monotonic_s, 100.)

    def test_future_radar_is_not_relabelled_and_retains_original_receive_time(self):
        for asynchronous in (True, False):
            with self.subTest(asynchronous=asynchronous):
                rig = self.rig(asynchronous=asynchronous)
                with patch("sensor_runtime.time.monotonic", return_value=499.):
                    rig._queue_radar(SimpleNamespace(frame=101, timestamp=2.525))
                first = self.read(rig)
                self.assertIsNone(first.radar_measurement)
                self.assertIsNone(first.radar_received_monotonic_s)
                rig.world.tick = lambda: 101
                second = self.read(rig, frame=101)
                self.assertEqual(second.radar_measurement.frame, 101)
                self.assertEqual(second.radar_received_monotonic_s, 499.)

    def test_missing_or_disabled_radar_has_no_fabricated_receive_timestamp(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                sample = self.read(self.rig(enabled=enabled))
                self.assertIsNone(sample.radar_measurement)
                self.assertIsNone(sample.radar_received_monotonic_s)

    def test_spawn_connects_radar_callback_to_timestamped_queue(self):
        rig = self.rig()
        for prefix in ("CAM", "LIDAR", "RADAR"):
            for axis in ("X", "Y", "Z", "ROLL", "PITCH", "YAW"):
                setattr(rig.cfg, f"{prefix}_{axis}", 0.)
        rig.cfg.FPS = 40
        rig.world = Mock()
        camera, lidar, radar = Mock(), Mock(), Mock()
        rig.world.spawn_actor.side_effect = [camera, lidar, radar]
        with patch("sensor_runtime.configure_camera_blueprint"), \
                patch("sensor_runtime.configure_lidar_blueprint"), \
                patch("sensor_runtime.configure_radar_blueprint"), \
                patch("sensor_runtime.CollisionSensor"):
            rig.spawn()
        data = SimpleNamespace(frame=100, timestamp=2.5)
        with patch("sensor_runtime.time.monotonic", return_value=499.95):
            radar.listen.call_args.args[0](data)
        sample = self.read(rig)
        self.assertIs(sample.radar_measurement, data)
        self.assertEqual(sample.radar_received_monotonic_s, 499.95)


if __name__ == "__main__":
    unittest.main()
