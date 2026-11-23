"""Regression: telemetry CSV phải nằm trong logs/telemetry/, không rải ở thư mục gốc.

Trước đây TelemetryLogger mở `telemetry_log_<ts>.csv` theo thư mục hiện hành,
nên mỗi lần chạy chinh.py lại thêm một file CSV vào project root.
"""

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "modules")):
    if path not in sys.path:
        sys.path.insert(0, path)

import config as cfg
from data_logger import TelemetryLogger


def _vec(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


class TelemetryLoggerTests(unittest.TestCase):
    def test_csv_is_created_inside_the_given_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "telemetry")
            logger = TelemetryLogger(directory=target)
            try:
                vehicle = SimpleNamespace(get_location=lambda: _vec(1.0, 2.0),
                                          get_velocity=lambda: _vec(10.0, 0.0))
                self.assertAlmostEqual(logger.log(vehicle, 3), 36.0)
            finally:
                logger.close()
            self.assertEqual(os.path.dirname(os.path.abspath(logger.filename)),
                             os.path.abspath(target))
            self.assertTrue(os.path.basename(logger.filename).startswith("telemetry_log_"))
            with open(logger.filename, encoding="utf-8") as stream:
                rows = stream.read().splitlines()
            self.assertEqual(rows[0], "Timestamp,Location_X,Location_Y,Speed_kmh,Objects_Tracked")
            self.assertEqual(len(rows), 2)

    def test_default_directory_is_logs_telemetry_not_cwd(self):
        self.assertEqual(Path(cfg.TELEMETRY_DIR), Path("logs/telemetry"))
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                logger = TelemetryLogger()
                logger.close()
                self.assertEqual(os.listdir(tmp), ["logs"])
                self.assertEqual(Path(logger.filename).parent, Path("logs/telemetry"))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
