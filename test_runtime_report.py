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
                      lane_source_counts={'map': 40})
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
