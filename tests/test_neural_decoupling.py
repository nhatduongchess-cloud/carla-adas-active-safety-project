"""Execute the real composition class with synthetic model dependencies only."""
import ast
import concurrent.futures
from contextlib import redirect_stdout
from dataclasses import fields
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from typing import Any, Optional, Tuple
import unittest
from unittest.mock import Mock

import numpy as np
import config
from modules.inference_scheduler import LatestFrameScheduler, usable_inference_payload
from modules.lane_source import LaneSourceArbiter, map_lane_estimate
from modules.perception_contracts import LaneEstimate
from modules.runtime_report import RuntimeReportData, write_runtime_report
from modules.pipeline_metrics import PipelineMetrics


def composition_class(dependencies):
    # Avoid importing/initializing any model, CARLA server or network dependency.
    tree = ast.parse(Path('modules/neural_perception.py').read_text(encoding='utf-8-sig'))
    body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    namespace = dict(Any=Any, Optional=Optional, Tuple=Tuple, time=time,
                     concurrent=concurrent, LatestFrameScheduler=LatestFrameScheduler,
                     usable_inference_payload=usable_inference_payload, LaneEstimate=LaneEstimate)
    namespace.update(dependencies)
    exec(compile(ast.Module(body, type_ignores=[]), 'neural_perception.py', 'exec'), namespace)
    return namespace['NeuralPerceptionRuntime']


class NeuralDecouplingTests(unittest.TestCase):
    def setUp(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.block_lane = False
        self.lane_error = False
        self.object_error = False
        self.object_images, self.lane_images = [], []
        def infer(image, frame, timestamp):
            self.lane_images.append(image)
            self.entered.set()
            if self.block_lane and not self.release.wait(4.):
                raise RuntimeError('test failed to release lane')
            if self.lane_error:
                raise RuntimeError('synthetic lane failure')
            return LaneEstimate(frame, timestamp, 'learned', confidence=.9,
                                centerline_m=[(0.,0.),(20.,0.)], lane_width_m=3.5,
                                offset_m=0., valid=True)
        def process(image, unused, frame):
            self.object_images.append(image)
            if self.object_error:
                raise RuntimeError('synthetic object failure')
            return image, [], {}
        self.adapter = NS(ready=True, infer=infer)
        dependencies = {
            'EnsembleVehicleTracker':Mock(return_value=NS(process=process, warmup=Mock())),
            'LidarProcessor':Mock(return_value=NS(extract_perception_obstacles=lambda pc: [])),
            'SensorFusion':Mock(return_value=NS(fuse=lambda detections, obstacles: [])),
            'LaneDetector':Mock(), 'GroundProjector':Mock(), 'UFLDv2Backend':Mock(),
            'LearnedLaneAdapter':Mock(return_value=self.adapter), 'LaneSourceArbiter':LaneSourceArbiter}
        self.cls = composition_class(dependencies)
        with redirect_stdout(io.StringIO()):
            self.runtime = self.cls(config)
        self.runtime.start_measurement()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.release.set()
        self.runtime.close()

    def submit(self, frame=10, timestamp=1., run_lane=True):
        image = np.zeros((2,2,3), dtype=np.uint8)
        self.runtime.submit(frame, timestamp, image, np.zeros((1,4)), run_lane)
        return image

    def wait_for(self, getter, predicate=lambda x: x is not None):
        deadline = time.monotonic()+1.
        while time.monotonic() < deadline:
            value = getter()
            if predicate(value):
                return value
            time.sleep(.001)
        self.fail('bounded test wait expired')

    def object(self, frame=10):
        return self.wait_for(lambda: self.runtime.poll(frame, 1.1, 150.),
                             lambda pair: pair[0] is not None and pair[0].frame_id == frame)

    def test_object_published_while_lane_is_still_held(self):
        self.block_lane = True
        self.submit()
        self.assertTrue(self.entered.wait(1.))
        result, product = self.object()
        self.assertIsNotNone(product)
        self.assertEqual(result.frame_id, 10)
        self.assertFalse(self.release.is_set())
        self.assertIsNone(product.get('lane'))

    def test_held_lane_never_builds_backlog_and_object_buffers_detached(self):
        self.block_lane = True
        for frame in range(10,16):
            image = self.submit(frame)
            image[:] = 255
            self.object(frame)
        lane = self.runtime.scheduler_stats()['lane']
        self.assertEqual(lane['submitted'], 1)
        self.assertEqual(lane['skipped_busy'], 5)
        self.assertTrue(lane['running'])
        self.assertFalse(lane['pending'])
        self.assertEqual(len(self.lane_images), 1)
        self.assertTrue(all(np.max(image) == 0 for image in self.object_images+self.lane_images))

    def test_lane_returns_original_frame_independent_of_new_object(self):
        self.block_lane = True
        self.submit(10)
        self.object(10)
        self.submit(11, run_lane=False)
        self.object(11)
        self.release.set()
        result, lane = self.wait_for(lambda: self.runtime.poll_lane(11,1.1,150.),
                                     lambda pair: pair[1] is not None)
        self.assertEqual((result.frame_id, lane.frame_id, lane.timestamp), (10,10,1.))
        self.runtime.record_lane_consumed(result,1.1)
        self.runtime.record_lane_consumed(result,1.1)
        self.assertEqual(self.runtime.scheduler_stats()['lane']['telemetry']['unique_consumed_frames'],1)
        self.assertIsNone(self.runtime.poll_lane(9,1.1,150.)[1])
        self.assertIsNone(self.runtime.poll_lane(11,1.151,150.)[1])

    def test_run_lane_false_and_absent_backend_do_not_start_lane(self):
        self.submit(run_lane=False)
        self.object()
        self.assertFalse(self.entered.is_set())
        self.runtime.close()
        self.adapter.ready = False
        with redirect_stdout(io.StringIO()):
            self.runtime = self.cls(config)
        self.runtime.start_measurement()
        self.submit(11)
        self.object(11)
        self.assertFalse(self.entered.is_set())
        self.assertIsNone(self.runtime._lane_scheduler)
        self.assertEqual(self.runtime.poll_lane(10,1.1,150.), (None,None))
        self.assertIsNone(self.runtime.scheduler_stats()['lane'])

    def test_lane_failure_keeps_objects_and_invalidates_lane_for_map_fallback(self):
        self.lane_error = True
        self.submit()
        self.assertIsNotNone(self.object()[1])
        result, lane = self.wait_for(lambda: self.runtime.poll_lane(10,1.1,150.),
                                     lambda pair: pair[0] is not None)
        self.assertIn('synthetic lane failure', result.error)
        self.assertFalse(lane.valid)
        fallback = map_lane_estimate(11,1.1,[(0.,0.),(20.,0.)])
        self.assertIs(self.runtime.lane_arbiter.select(lane,fallback,now_s=1.1,max_age_ms=150.),fallback)
        self.assertEqual(self.runtime.scheduler_stats()['failed'],0)
        self.assertEqual(self.runtime.scheduler_stats()['lane']['failed'],1)

    def test_object_failure_does_not_hide_independent_lane(self):
        self.object_error = True
        self.submit()
        result, product = self.object()
        self.assertIsNone(product)
        self.assertIn('synthetic object failure',result.error)
        lane_result, lane = self.wait_for(lambda: self.runtime.poll_lane(10,1.1,150.),
                                          lambda pair: pair[1] is not None)
        self.assertTrue(lane.valid)

    def test_duplicate_lane_poll_does_not_earn_five_observations_and_junction_wins(self):
        self.submit()
        _, lane = self.wait_for(lambda: self.runtime.poll_lane(10,1.1,150.),
                                lambda pair: pair[1] is not None)
        fallback = map_lane_estimate(20,1.1,[(0.,0.),(20.,0.)])
        arbiter = self.runtime.lane_arbiter
        for _ in range(8):
            self.assertIs(arbiter.select(lane,fallback,now_s=1.1,max_age_ms=150.),fallback)
        self.assertEqual(arbiter._valid_count,1)
        for frame in range(11,15):
            lane.frame_id = frame
            selected = arbiter.select(lane,fallback,now_s=1.1,max_age_ms=150.)
        self.assertIs(selected,lane)
        self.assertIs(arbiter.select(lane,fallback,at_junction=True,now_s=1.1,max_age_ms=150.),fallback)

    def test_close_is_bounded_reports_running_lane_and_disallows_new_work(self):
        self.block_lane = True
        self.submit()
        self.object()
        self.assertTrue(self.entered.wait(1.))
        started = time.monotonic()
        self.runtime.close(timeout=.02)
        self.assertLess(time.monotonic()-started,.3)
        stats = self.runtime.scheduler_stats()
        self.assertTrue(stats['lane']['thread_alive'])
        self.assertTrue(stats['lane']['telemetry']['window_closed'])
        with self.assertRaises(RuntimeError):
            self.submit(11)
        self.release.set()
        self.wait_for(lambda: self.runtime.scheduler_stats()['lane']['thread_alive'],lambda v:not v)

    def test_lane_failure_or_still_running_worker_cannot_pass_report(self):
        self.lane_error = True
        self.submit()
        self.object()
        self.wait_for(lambda:self.runtime.scheduler_stats()['lane']['failed'],lambda n:n==1)
        self.runtime.close()
        values = {f.name:None for f in fields(RuntimeReportData)}
        metrics = PipelineMetrics()
        metrics.record('safety_control',1.)
        metrics.record('neural_inference',1.)
        metrics.record_inference_age(1.)
        values.update(world=NS(get_map=lambda:NS(name='fixture')), frame_count=1, fps_ema=40.,
                      duration_s=.025, stop_reason='duration_reached', max_speed_kmh=0.,
                      distance_travelled_m=0., pipeline_metrics=metrics, perception_runtime=self.runtime,
                      health_monitor=NS(summary=lambda:{'availability':{'radar':1.}}),
                      lane_source_counts={'map':1},aeb_triggered=False)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder)/'report.json'
            self.assertFalse(write_runtime_report(str(path),config,RuntimeReportData(**values)))
            report = json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(report['criteria']['neural_inference_errors_zero'])
        self.assertTrue(report['criteria']['neural_workers_stopped'])
        stats = self.runtime.scheduler_stats()
        stats['lane']['failed'] = 0
        stats['lane']['thread_alive'] = True
        self.runtime.scheduler_stats = lambda:stats
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            path = Path(folder)/'report.json'
            self.assertFalse(write_runtime_report(str(path),config,RuntimeReportData(**values)))
            report = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(report['criteria']['neural_inference_errors_zero'])
        self.assertFalse(report['criteria']['neural_workers_stopped'])

    def test_optional_lane_submission_racing_close_does_not_poison_object(self):
        entered, release = threading.Event(), threading.Event()
        scheduler = self.runtime._lane_scheduler
        original = scheduler.submit
        def paused(*args,**kwargs):
            entered.set()
            if not release.wait(2.):
                raise RuntimeError('test failed to release dispatch')
            return original(*args,**kwargs)
        scheduler.submit = paused
        self.addCleanup(release.set)
        self.submit()
        self.assertTrue(entered.wait(1.))
        self.runtime.close(timeout=.01)
        release.set()
        self.wait_for(lambda:self.runtime._scheduler.stats()['thread_alive'],lambda v:not v)
        self.assertEqual(self.runtime._scheduler.stats()['failed'],0)
        self.assertEqual(self.runtime._scheduler.stats()['completed'],1)
        self.assertEqual(scheduler.stats()['skipped_closed'],1)
        self.assertEqual(scheduler.stats()['submitted'],0)

    def test_idle_only_rejects_pending_even_before_worker_starts(self):
        scheduler = self.runtime._lane_scheduler
        with scheduler._condition:  # prevent worker from dequeuing this pending request
            self.assertTrue(scheduler.submit(10,1.,(np.zeros((2,2,3)),1.),only_if_idle=True))
            self.assertFalse(scheduler.submit(11,1.,(np.zeros((2,2,3)),1.),only_if_idle=True))
            self.assertTrue(scheduler.stats()['pending'])
            self.assertFalse(scheduler.stats()['running'])
        self.wait_for(lambda:scheduler.stats()['completed'],lambda n:n==1)
        self.assertEqual(scheduler.stats()['skipped_busy'],1)

    def test_orchestrator_polls_lane_outside_fresh_object_branch(self):
        tree = ast.parse(Path('chinh.py').read_text(encoding='utf-8-sig'))
        calls = [(node,node.value) for node in ast.walk(tree)
                 if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                 and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='poll_lane']
        self.assertEqual(len(calls),1)
        statement,_ = calls[0]
        parent = next(node for node in ast.walk(tree) if isinstance(node,ast.While)
                      and statement in node.body)
        self.assertTrue(any(isinstance(node,ast.If) and isinstance(node.test,ast.Name)
                            and node.test.id=='fresh_inference' for node in parent.body))


if __name__ == '__main__':
    unittest.main()
