"""W1.4: honest completion/consumption accounting without changing admission."""
import ast
from contextlib import ExitStack, contextmanager
from dataclasses import fields
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.inference_scheduler import LatestFrameScheduler
from modules.pipeline_metrics import PipelineMetrics
from modules.runtime_report import RuntimeReportData, write_runtime_report


class InferenceTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.
        self.patch = patch('modules.inference_scheduler.time', SimpleNamespace(
            monotonic=lambda: self.clock, perf_counter=lambda: self.clock))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.scheduler = LatestFrameScheduler(self.worker, max_samples=2)
        self.addCleanup(self.scheduler.close)
        self.scheduler.start_measurement()

    def worker(self, payload, frame):
        self.clock += .010
        if payload == 'error':
            raise RuntimeError('synthetic worker failure')
        return {'detections': [], 'stage_latency_ms': {'object_detection': 8.}}

    def finish(self, frame=10, timestamp=100., payload=None):
        expected = self.scheduler.stats()['completed'] + self.scheduler.stats()['failed'] + 1
        self.scheduler.submit(frame, timestamp, payload)
        deadline = time.monotonic() + 2.
        while time.monotonic() < deadline:
            state = self.scheduler.stats()
            if state['completed'] + state['failed'] == expected:
                return
            time.sleep(.001)
        self.fail('worker did not publish within test deadline')

    def telemetry(self):
        return self.scheduler.stats()['telemetry']

    def test_stale_completion_keeps_stage_latency_and_no_consumed_fps(self):
        self.finish()
        self.clock = 100.2
        for _ in range(5):
            self.assertIsNone(self.scheduler.latest(11, self.clock, 150.))
        stats = self.telemetry()
        self.assertEqual(stats['first_poll_outcomes'], {'stale': 1})
        self.assertEqual(stats['stages']['object_detection']['total_samples'], 1)
        self.assertEqual(stats['stages']['neural_inference']['total_samples'], 1)
        self.assertAlmostEqual(stats['stages']['queue_wait']['p50_ms'], 0.)
        self.assertEqual(stats['unique_consumed_frames'], 0)
        self.assertEqual(stats['consumed_output_fps'], 0.)

    def test_repeated_poll_and_ack_do_not_inflate_unique_output(self):
        self.finish()
        result = self.scheduler.latest(10, 100.1)
        for _ in range(4):
            self.assertIs(self.scheduler.latest(10, 100.1), result)
            self.scheduler.record_consumed(result, 100.1)
        self.clock = 101.
        stats = self.telemetry()
        self.assertEqual(stats['first_poll_outcomes'], {'eligible': 1})
        self.assertEqual(stats['unique_consumed_frames'], 1)
        self.assertEqual(stats['consumed_output_fps'], 1.)
        self.assertEqual(stats['stages']['consumed_age']['samples'], 1)

    def test_future_first_poll_can_become_eligible_without_rewriting_history(self):
        self.finish()
        self.assertIsNone(self.scheduler.latest(9, 100.1))
        result = self.scheduler.latest(10, 100.1)
        self.assertIsNotNone(result)
        self.scheduler.record_consumed(result, 100.1)
        self.assertEqual(self.telemetry()['first_poll_outcomes'], {'future_frame': 1})
        self.assertEqual(self.telemetry()['unique_consumed_frames'], 1)

    def test_worker_error_is_counted_without_consumption(self):
        self.finish(payload='error')
        result = self.scheduler.latest(10, 100.1)
        self.assertIn('synthetic worker failure', result.error)
        self.scheduler.record_consumed(result, 100.1)
        stats = self.telemetry()
        self.assertEqual(stats['first_poll_outcomes'], {'worker_error': 1})
        self.assertEqual(stats['finished_requests'], 1)
        self.assertEqual(stats['unique_completed_frames'], 0)
        self.assertEqual(stats['unique_consumed_frames'], 0)

    def test_completed_overwrite_and_bounded_samples_have_lifetime_denominator(self):
        for frame in range(10, 14):
            self.finish(frame, self.clock)
        stats = self.telemetry()
        self.assertEqual(stats['finished_requests'], 4)
        self.assertEqual(stats['overwritten_before_poll'], 3)
        self.assertEqual(stats['unpolled_latest'], 1)
        self.assertEqual(stats['stages']['neural_inference']['samples'], 2)
        self.assertEqual(stats['stages']['neural_inference']['total_samples'], 4)
        json.dumps(stats, allow_nan=False)

    def test_duplicate_and_out_of_order_completions_not_unique_frames(self):
        for frame in (10, 10, 9, 11):
            self.finish(frame, self.clock)
            result = self.scheduler.latest(12, self.clock)
            self.scheduler.record_consumed(result, self.clock)
        stats = self.telemetry()
        self.assertEqual(stats['finished_requests'], 4)
        self.assertEqual(stats['unique_completed_frames'], 2)
        self.assertEqual(stats['unique_consumed_frames'], 2)
        self.assertEqual(stats['nonincreasing_completed_frames'], 2)

    def test_invalid_clock_has_explicit_reason_and_does_not_disable_guard(self):
        self.finish(timestamp=math.nan)
        self.assertIsNone(self.scheduler.latest(10, 100.1))
        self.assertEqual(self.telemetry()['first_poll_outcomes'], {'invalid_clock': 1})

    def test_stop_freezes_window_and_excludes_shutdown_work(self):
        entered, release = threading.Event(), threading.Event()
        original = self.scheduler.worker
        def held(payload, frame):
            entered.set()
            if not release.wait(2.):
                raise AssertionError('test did not release worker')
            return original(payload, frame)
        self.scheduler.worker = held
        self.addCleanup(release.set)
        self.scheduler.submit(10, 100., None)
        self.assertTrue(entered.wait(2.))
        self.clock = 101.
        self.scheduler.stop_measurement()
        release.set()
        self.scheduler.close()
        self.clock = 110.
        stats = self.telemetry()
        self.assertEqual(stats['duration_s'], 1.)
        self.assertEqual(stats['finished_requests'], 0)
        self.assertEqual(stats['finished_after_window'], 1)
        self.assertEqual(stats['stages'], {})
        self.assertEqual(self.scheduler.stats()['completed'], 1)

    def test_actual_orchestrator_freezes_before_sync_teardown_on_exception(self):
        # Execute only the actual with-items/start/callback wiring, not CARLA/models.
        tree = ast.parse(Path('chinh.py').read_text(encoding='utf-8-sig'))
        context = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
                       and any(isinstance(i.context_expr, ast.Name)
                               and i.context_expr.id == 'runtime_context' for i in n.items))
        wiring = [n for n in context.body if isinstance(n, ast.Expr)
                  and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr in ('start_measurement', 'callback')]
        self.assertEqual(len(wiring), 2)
        def work():
            self.clock = 101.
            raise RuntimeError('loop fault')
        @contextmanager
        def sync_world():
            try:
                yield
            finally:
                self.clock = 110.  # simulated slow world restoration
        context.body = wiring + [ast.Expr(ast.Call(ast.Name('work', ast.Load()), [], []))]
        module = ast.fix_missing_locations(ast.Module([context], type_ignores=[]))
        with self.assertRaisesRegex(RuntimeError, 'loop fault'):
            exec(compile(module, 'chinh-measurement-wiring', 'exec'), {
                'runtime_context': sync_world(), 'ExitStack': ExitStack,
                'perception_runtime': self.scheduler, 'work': work})
        self.assertEqual(self.telemetry()['duration_s'], 1.)

    def test_report_exposes_completed_work_without_turning_missing_output_into_pass(self):
        self.finish()
        self.clock = 100.2
        self.assertIsNone(self.scheduler.latest(11, self.clock))
        self.scheduler.stop_measurement()
        runtime = SimpleNamespace(
            scheduler_stats=self.scheduler.stats,
            lane_adapter=SimpleNamespace(ready=False), lane_backend_error=None)
        values = {f.name: None for f in fields(RuntimeReportData)}
        values.update(world=SimpleNamespace(get_map=lambda: SimpleNamespace(name='fixture')),
                      frame_count=1, fps_ema=40., duration_s=.025, stop_reason='duration_reached',
                      max_speed_kmh=0., distance_travelled_m=0.,
                      pipeline_metrics=PipelineMetrics(), perception_runtime=runtime,
                      health_monitor=SimpleNamespace(summary=lambda: {'availability':{'radar':1.}}),
                      lane_source_counts={'map':1}, aeb_triggered=False)
        import config as cfg
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder)/'report.json'
            passed = write_runtime_report(str(report_path), cfg, RuntimeReportData(**values))
            report = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertFalse(passed)
        self.assertFalse(report['criteria']['perception_p95_le_50ms'])
        measured = report['pipeline']['inference_scheduler']['telemetry']
        self.assertEqual(measured['finished_requests'], 1)
        self.assertEqual(measured['first_poll_outcomes'], {'stale': 1})
        self.assertEqual(measured['unique_consumed_frames'], 0)
        self.assertFalse(report['safety']['aeb_triggered'])
        json.dumps(report, allow_nan=False)

    def test_pending_replacement_and_queue_wait_remain_distinct(self):
        entered, release = threading.Event(), threading.Event()
        original = self.scheduler.worker
        def held(payload, frame):
            if frame == 10:
                entered.set()
                if not release.wait(2.):
                    raise AssertionError('test did not release worker')
            return original(payload, frame)
        self.scheduler.worker = held
        self.addCleanup(release.set)
        self.scheduler.submit(10, 100., None)
        self.assertTrue(entered.wait(2.))
        self.clock = 100.1
        self.scheduler.submit(11, 100.1, None)
        self.scheduler.submit(12, 100.1, None)
        self.clock = 100.2
        release.set()
        deadline = time.monotonic() + 2.
        while self.scheduler.stats()['completed'] != 2 and time.monotonic() < deadline:
            time.sleep(.001)
        stats = self.scheduler.stats()
        self.assertEqual(stats['completed'], 2)
        self.assertEqual(stats['replaced'], 1)
        self.assertAlmostEqual(stats['telemetry']['stages']['queue_wait']['p99_ms'], 110.)
        self.assertEqual(self.scheduler.latest(12, self.clock).frame_id, 12)


class StageAccountingTests(unittest.TestCase):
    def test_window_lifetime_and_invalid_samples_are_distinct(self):
        metrics = PipelineMetrics({'worker': 50.}, max_samples=2)
        for value in (60., 10., 20., math.nan, math.inf, -1., None):
            metrics.record('worker', value)
        stage = metrics.summary()['stages']['worker']
        self.assertEqual(stage['samples'], 2)
        self.assertEqual(stage['total_samples'], 3)
        self.assertEqual(stage['deadline_misses'], 1)
        self.assertEqual(stage['window_deadline_misses'], 0)
        self.assertEqual(stage['invalid_samples'], 4)
        json.dumps(metrics.summary(), allow_nan=False)

    def test_nonpositive_capacity_is_rejected(self):
        for capacity in (0, -1):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                PipelineMetrics(max_samples=capacity)


if __name__ == '__main__':
    unittest.main()
