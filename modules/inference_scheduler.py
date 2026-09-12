"""Single-worker, latest-frame-only scheduler for bounded GPU inference.

There is exactly one pending slot.  A newly submitted frame atomically replaces
an older request that has not started, preventing latency from accumulating.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Optional

try:
    from perception_contracts import TaggedInferenceResult, validate_frame_age
    from pipeline_metrics import PipelineMetrics
except ImportError:
    from modules.perception_contracts import TaggedInferenceResult, validate_frame_age
    from modules.pipeline_metrics import PipelineMetrics


@dataclass
class InferenceRequest:
    frame_id: int
    timestamp: float
    payload: Any
    enqueued_at: float = field(default_factory=lambda: time.monotonic())


def usable_inference_payload(result: Optional[TaggedInferenceResult]):
    """Return payload only for a successful result already admitted by age gating."""
    if result is None or result.error is not None:
        return None
    return result.payload


class LatestFrameScheduler:
    def __init__(self, worker: Callable[[Any, int], Any], name="gpu-inference",
                 *, max_samples=256, deadline_ms=50.0):
        self.worker = worker
        self.name = name
        self._condition = threading.Condition()
        self._pending: Optional[InferenceRequest] = None
        self._latest: Optional[TaggedInferenceResult] = None
        self._closed = False
        self.submitted = 0
        self.replaced = 0
        self.completed = 0
        self.failed = 0
        self.skipped_busy = 0
        self.skipped_closed = 0
        self._running = False
        self._metrics = PipelineMetrics({"neural_inference": deadline_ms}, max_samples)
        self._counts = Counter()
        self._first_poll_outcomes = Counter()
        self._latest_polled = False
        self._highest_completed_frame = None
        self._highest_consumed_frame = None
        self._measurement_started = time.monotonic()
        self._measurement_scope = "scheduler_lifetime"
        self._frozen_telemetry = None
        self._finished_after_window = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, timestamp: float, payload: Any,
               *, only_if_idle=False) -> bool:
        """Offer optional idle-only work, or replace the normal pending request."""
        request = InferenceRequest(int(frame_id), float(timestamp), payload)
        with self._condition:
            if self._closed:
                if only_if_idle:
                    # Optional work may race bounded shutdown; never queue it or
                    # turn a successful sibling object request into a worker error.
                    self.skipped_closed += 1
                    return False
                raise RuntimeError(f"{self.name} is closed")
            if only_if_idle and (self._running or self._pending is not None):
                self.skipped_busy += 1
                return False
            if self._pending is not None:
                self.replaced += 1
            self._pending = request
            self.submitted += 1
            self._condition.notify()
            return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._pending is None and self._closed:
                    return
                request = self._pending
                self._pending = None
                self._running = True
            started = time.perf_counter()
            dequeued_at = time.monotonic()
            try:
                payload = self.worker(request.payload, request.frame_id)
                result = TaggedInferenceResult(
                    request.frame_id, request.timestamp, time.perf_counter(), payload,
                    (time.perf_counter() - started) * 1000.0)
            except Exception as exc:
                result = TaggedInferenceResult(
                    request.frame_id, request.timestamp, time.perf_counter(), None,
                    (time.perf_counter() - started) * 1000.0,
                    error=f"{type(exc).__name__}: {exc}")
            with self._condition:
                if result.error is None:
                    self.completed += 1
                else:
                    self.failed += 1
                if self._frozen_telemetry is None:
                    self._counts['finished_requests'] += 1
                    if self._latest is not None and not self._latest_polled:
                        self._counts['overwritten_before_poll'] += 1
                    if result.error is None and result.payload is not None:
                        if (self._highest_completed_frame is None
                                or result.frame_id > self._highest_completed_frame):
                            self._highest_completed_frame = result.frame_id
                            self._counts['unique_completed_frames'] += 1
                        else:
                            self._counts['nonincreasing_completed_frames'] += 1
                    self._metrics.record('queue_wait', (dequeued_at-request.enqueued_at)*1000.)
                    self._metrics.record('neural_inference', result.latency_ms)
                    self._metrics.record('submission_to_completion',
                                         (time.monotonic()-request.enqueued_at)*1000.)
                    # Fixed stage names keep telemetry bounded; never retain images/payloads.
                    stages = (result.payload.get('stage_latency_ms', {})
                              if isinstance(result.payload, dict) else {})
                    if isinstance(stages, dict):
                        for stage in ('object_detection', 'lidar_fusion', 'learned_lane'):
                            if stages.get(stage) is not None:
                                self._metrics.record(stage, stages[stage])
                else:
                    self._finished_after_window += 1
                self._latest = result
                self._latest_polled = False
                self._running = False

    def latest(self, current_frame_id: Optional[int] = None, now_s: Optional[float] = None,
               max_age_ms: float = 150.0) -> Optional[TaggedInferenceResult]:
        with self._condition:
            result = self._latest
            if result is None:
                return None
            admitted = result
            outcome = ('worker_error' if result.error is not None else
                       'empty_payload' if result.payload is None else 'eligible')
            try:
                validate_frame_age(
                    result.frame_id, result.timestamp,
                    result.frame_id if current_frame_id is None else current_frame_id,
                    time.monotonic() if now_s is None else now_s, max_age_ms)
            except (TypeError, ValueError, OverflowError) as error:
                admitted = None
                if result.error is None:
                    outcome = getattr(error, 'reason', 'invalid_input')
            if self._frozen_telemetry is None and not self._latest_polled:
                self._first_poll_outcomes[outcome] += 1
                self._latest_polled = True
            return admitted

    def record_consumed(self, result, now_s):
        """Consumer acknowledgement, not admission. Never changes a control payload."""
        with self._condition:
            if (self._frozen_telemetry is not None or result is None
                    or result.error is not None or result.payload is None):
                return
            # CARLA frame IDs increase within one runtime/episode. Do not inflate
            # unique FPS for duplicate/out-of-order acknowledgements; do not drop their work.
            if (self._highest_consumed_frame is not None
                    and result.frame_id <= self._highest_consumed_frame):
                return
            if self._metrics.record('consumed_age', result.age_ms(now_s)):
                self._highest_consumed_frame = result.frame_id
                self._counts['unique_consumed_frames'] += 1

    def start_measurement(self):
        """Exclude model/sensor startup; call once before submitting runtime work."""
        with self._condition:
            if self.submitted or self._closed or self._frozen_telemetry is not None:
                raise RuntimeError('measurement must start before inference submissions')
            self._measurement_started = time.monotonic()
            self._measurement_scope = 'runtime_loop'

    def _telemetry_summary(self, now_s):
        duration = max(0., now_s-self._measurement_started)
        return {
            'schema_version': 1, 'scope': self._measurement_scope,
            'duration_s': duration, 'sample_capacity': self._metrics.max_samples,
            **{key: self._counts[key] for key in (
                'finished_requests', 'unique_completed_frames', 'nonincreasing_completed_frames',
                'unique_consumed_frames', 'overwritten_before_poll')},
            'first_poll_outcomes': dict(self._first_poll_outcomes),
            'unpolled_latest': int(self._latest is not None and not self._latest_polled),
            'completed_output_fps': self._counts['unique_completed_frames']/duration if duration else None,
            'consumed_output_fps': self._counts['unique_consumed_frames']/duration if duration else None,
            'stages': self._metrics.summary()['stages'],
        }

    def stop_measurement(self):
        """Freeze the runtime window before cleanup/draining queued work."""
        with self._condition:
            if self._frozen_telemetry is None:
                self._frozen_telemetry = self._telemetry_summary(time.monotonic())

    def stats(self) -> dict:
        with self._condition:
            return {"submitted": self.submitted, "completed": self.completed,
                    "failed": self.failed, "replaced": self.replaced,
                    "pending": self._pending is not None,
                    "running": self._running, "skipped_busy": self.skipped_busy,
                    "skipped_closed": self.skipped_closed,
                    "thread_alive": self._thread.is_alive(),
                    "telemetry": {
                        **deepcopy(self._frozen_telemetry if self._frozen_telemetry is not None
                                   else self._telemetry_summary(time.monotonic())),
                        "window_closed": self._frozen_telemetry is not None,
                        "finished_after_window": self._finished_after_window,
                    }}

    def close(self, timeout=2.0) -> None:
        self.stop_measurement()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
