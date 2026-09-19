"""Offline checks for camera/LiDAR pairing-skew accounting (B03/B04).

A valid at-or-before async frame raises no frame_error, so temporal skew between
the camera and the LiDAR reference frame used to be invisible. These tests pin
that the skew is now measured and summarised.
"""
import unittest

from modules.sensor_sync import SensorSyncStats


class PairingSkewTests(unittest.TestCase):
    def test_exact_pairing_reports_zero_skew(self):
        stats = SensorSyncStats()
        for _ in range(5):
            stats.record_pairing_skew(0)
        summary = stats.summary()
        self.assertEqual(summary["pairing_samples"], 5)
        self.assertEqual(summary["pairing_skew_max_frames"], 0)
        self.assertEqual(summary["pairing_skew_p95_frames"], 0)

    def test_async_skew_is_measured_not_hidden(self):
        stats = SensorSyncStats()
        for skew in (0, 1, 2, 1, 3):  # cached-RGB reuse grows the skew
            stats.record_pairing_skew(skew)
        summary = stats.summary()
        self.assertEqual(summary["pairing_samples"], 5)
        self.assertEqual(summary["pairing_skew_max_frames"], 3)
        self.assertGreaterEqual(summary["pairing_skew_p95_frames"], 2)
        # frame_errors stays 0 for valid at-or-before frames; skew is the signal.
        self.assertEqual(summary["frame_errors"], 0)

    def test_negative_skew_is_clamped(self):
        stats = SensorSyncStats()
        stats.record_pairing_skew(-4)
        self.assertEqual(stats.summary()["pairing_skew_max_frames"], 0)

    def test_no_samples_is_zero_not_error(self):
        summary = SensorSyncStats().summary()
        self.assertEqual(summary["pairing_samples"], 0)
        self.assertEqual(summary["pairing_skew_max_frames"], 0)
        self.assertEqual(summary["pairing_skew_p95_frames"], 0.0)


if __name__ == "__main__":
    unittest.main()
