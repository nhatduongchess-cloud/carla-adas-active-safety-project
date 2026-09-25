"""Fail-closed finite runtime evidence; synthetic, not simulator qualification."""
from contextlib import redirect_stdout
from dataclasses import fields
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import config
from modules.runtime_report import RuntimeReportData, write_runtime_report


class RuntimeReportTests(unittest.TestCase):
    def fixture(self):
        pipeline = {'stages': {'safety_control': {'p99_ms': 1.},
                               'neural_inference': {'p95_ms': 1.}},
                    'inference_age_p95_ms': 1., 'gpu_peak_mb': 100.,
                    'gpu_device_peak_used_mb': 100.}
        scheduler = {'failed': 0, 'thread_alive': False, 'lane': None}
        values = {f.name: None for f in fields(RuntimeReportData)}
        values.update(world=NS(get_map=lambda: NS(name='fixture')), town_name='fixture',
                      frame_count=40, fps_ema=40., duration_s=1., stop_reason='duration_reached',
                      max_speed_kmh=20., distance_travelled_m=5.,
                      pipeline_metrics=NS(summary=lambda: pipeline),
                      perception_runtime=NS(scheduler_stats=lambda: scheduler,
                                            lane_adapter=NS(ready=False), lane_backend_error=None),
                      health_monitor=NS(summary=lambda: {'availability': {'radar': 1.}}),
                      collision_sensor=NS(count=0, history=[]), sensor_sync_stats=NS(frame_errors=0),
                      radar_sync_stats=NS(frame_errors=0), cleanup_summary={'verified': True},
                      lane_source_counts={'map': 40},
                      # The rate gate reads the measured control window, not
                      # fps_ema, so a passing fixture must carry a measurement.
                      control_timing={'status': 'MEASURED', 'delivered_control_hz': 40.0})
        return RuntimeReportData(**values), pipeline

    def render(self, data):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder) / 'report.json'
            passed = write_runtime_report(str(path), config, data)
            text = path.read_text(encoding='utf-8')
            report = json.loads(text)
            json.dumps(report, allow_nan=False)
        return passed, report

    def test_complete_fixture_passes_legacy_runtime_gate_only(self):
        data, _ = self.fixture()
        self.assertTrue(self.render(data)[0])

    def test_missing_collision_sync_or_cleanup_evidence_fails(self):
        for field in ('collision_sensor', 'sensor_sync_stats', 'cleanup_summary'):
            data, _ = self.fixture()
            setattr(data, field, None)
            self.assertFalse(self.render(data)[0], field)

    def test_invalid_required_numbers_never_pass_or_break_json(self):
        for value in (float('-inf'), float('inf'), float('nan'), -1.):
            for stage, metric in (('safety_control', 'p99_ms'), ('neural_inference', 'p95_ms')):
                data, pipeline = self.fixture()
                pipeline['stages'][stage][metric] = value
                with self.subTest(value=value, stage=stage):
                    self.assertFalse(self.render(data)[0])

    def test_missing_memory_and_zero_frames_fail(self):
        data, pipeline = self.fixture()
        pipeline['gpu_device_peak_used_mb'] = None
        self.assertFalse(self.render(data)[0])
        data, _ = self.fixture()
        data.duration_s = None
        data.frame_count = 0
        self.assertFalse(self.render(data)[0])

    def test_rate_gate_uses_measured_control_rate_not_hud_ema(self):
        """The published demo: fps_ema 44.9, frames / wall 20.0."""
        data, _ = self.fixture()
        data.fps_ema = 44.9
        data.control_timing = {'status': 'MEASURED', 'delivered_control_hz': 20.0}
        passed, report = self.render(data)
        self.assertFalse(passed)
        self.assertFalse(report['criteria']['delivered_control_hz_ge_95pct_target'])
        self.assertEqual(report['run']['hud_fps_ema'], 44.9)

    def test_unmeasured_control_rate_fails_rather_than_passing_on_ema(self):
        for timing in (None, {'status': 'NOT_EVALUATED', 'delivered_control_hz': None},
                       {'status': 'MEASURED', 'delivered_control_hz': float('nan')},
                       {'status': 'MEASURED', 'delivered_control_hz': True}):
            with self.subTest(timing=timing):
                data, _ = self.fixture()
                data.control_timing = timing
                self.assertFalse(self.render(data)[0])

    def test_disabled_radar_is_not_applicable_rather_than_available(self):
        """A disabled radar used to report 100% availability and pass the
        radar criterion. It now reports None and is listed as not applicable."""
        data, _ = self.fixture()
        data.health_monitor = NS(summary=lambda: {
            'enabled': {'camera': True, 'lidar': True, 'radar': False},
            'availability': {'camera': 1., 'lidar': 1., 'radar': None}})
        passed, report = self.render(data)
        self.assertTrue(passed)
        self.assertIsNone(report['criteria']['radar_availability_ge_99_5pct'])
        self.assertEqual(report['criteria_not_applicable'], ['radar_availability_ge_99_5pct'])

    def test_enabled_radar_with_no_availability_fails_rather_than_being_skipped(self):
        """Only a criterion that is explicitly not applicable may be skipped."""
        data, _ = self.fixture()
        data.health_monitor = NS(summary=lambda: {
            'enabled': {'radar': True}, 'availability': {'radar': None}})
        passed, report = self.render(data)
        self.assertFalse(passed)
        self.assertEqual(report['criteria_not_applicable'], [])

    def test_reporting_after_server_loss_uses_cached_map(self):
        data, _ = self.fixture()
        data.world = None
        data.run_error = 'native server unavailable'
        passed, report = self.render(data)
        self.assertFalse(passed)
        self.assertEqual(report['run']['town'], 'fixture')
        self.assertEqual(report['error'], 'native server unavailable')


if __name__ == '__main__':
    unittest.main()
