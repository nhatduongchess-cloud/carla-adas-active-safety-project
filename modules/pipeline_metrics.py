"""Stage-level latency, deadline, inference-age, and GPU-memory telemetry."""

from collections import defaultdict
from contextlib import contextmanager
import math
import time


class PipelineMetrics:
    def __init__(self, deadlines_ms=None, max_samples=20000):
        self.deadlines_ms = dict(deadlines_ms or {})
        for name, value in self.deadlines_ms.items():
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'Invalid deadline for {name}') from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid deadline for {name}')
            self.deadlines_ms[name] = value
        self.max_samples = int(max_samples)
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.samples = defaultdict(list)
        self.total_samples = defaultdict(int)
        self.invalid_samples = defaultdict(int)
        self.deadline_misses = defaultdict(int)
        self.inference_age_ms = []
        self.inference_age_total_samples = 0
        self.inference_age_invalid_samples = 0
        self.gpu_peak_mb = None
        # Device-wide usage includes CARLA/Unreal, unlike max_memory_allocated()
        # which only sees tensors owned by this PyTorch process.
        self.gpu_device_peak_used_mb = None
        self.gpu_device_total_mb = None
        self.gpu_tensor_samples = 0
        self.gpu_device_samples = 0
        self.gpu_sampling_errors = 0
        self.gpu_last_error = None

    def record(self, stage, latency_ms):
        stage = str(stage)
        try:
            value = float(latency_ms)
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        if not math.isfinite(value) or value < 0.0:
            self.invalid_samples[stage] += 1
            return False
        values = self.samples[stage]
        self.total_samples[stage] += 1
        values.append(value)
        if len(values) > self.max_samples:
            del values[:len(values) - self.max_samples]
        deadline = self.deadlines_ms.get(str(stage))
        if deadline is not None and value > float(deadline):
            self.deadline_misses[str(stage)] += 1
        return True

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - started) * 1000.0)

    def record_inference_age(self, age_ms):
        try:
            value = float(age_ms)
        except (TypeError, ValueError, OverflowError):
            value = math.nan
        if isinstance(age_ms, bool) or not math.isfinite(value) or value < 0:
            self.inference_age_invalid_samples += 1
            return False
        self.inference_age_total_samples += 1
        self.inference_age_ms.append(value)
        if len(self.inference_age_ms) > self.max_samples:
            del self.inference_age_ms[:len(self.inference_age_ms) - self.max_samples]
        return True

    def sample_gpu_memory(self):
        try:
            import torch
            if torch.cuda.is_available():
                tensor_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                if not math.isfinite(tensor_mb) or tensor_mb < 0:
                    raise ValueError('invalid tensor memory sample')
                self.gpu_peak_mb = max(self.gpu_peak_mb or 0., tensor_mb)
                self.gpu_tensor_samples += 1
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if not (math.isfinite(free_bytes) and math.isfinite(total_bytes)
                        and 0 <= free_bytes <= total_bytes and total_bytes > 0):
                    raise ValueError('invalid device memory sample')
                device_used_mb = float(total_bytes - free_bytes) / (1024.0 * 1024.0)
                self.gpu_device_peak_used_mb = max(
                    self.gpu_device_peak_used_mb or 0., device_used_mb)
                self.gpu_device_total_mb = float(total_bytes) / (1024.0 * 1024.0)
                self.gpu_device_samples += 1
                self.gpu_last_error = None
            else:
                self.gpu_last_error = 'CUDA unavailable; memory not measured'
        except Exception as error:
            self.gpu_sampling_errors += 1
            self.gpu_last_error = f'{type(error).__name__}: {error}'
        return self.gpu_peak_mb

    @staticmethod
    def _percentile(values, q):
        if not values:
            return None
        xs = sorted(values)
        index = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return round(xs[index], 3)

    def summary(self):
        stages = {}
        for name in self.samples.keys() | self.invalid_samples.keys():
            values = self.samples.get(name, [])
            deadline = self.deadlines_ms.get(name)
            stages[name] = {
                "samples": len(values),
                "total_samples": self.total_samples.get(name, 0),
                "invalid_samples": self.invalid_samples.get(name, 0),
                "p50_ms": self._percentile(values, 0.50),
                "p95_ms": self._percentile(values, 0.95),
                "p99_ms": self._percentile(values, 0.99),
                "deadline_ms": self.deadlines_ms.get(name),
                "deadline_misses": self.deadline_misses.get(name, 0),
                "window_deadline_misses": (sum(v > float(deadline) for v in values)
                                           if deadline is not None else 0),
                "deadline_miss_fraction": (self.deadline_misses.get(name, 0) / self.total_samples[name]
                                           if deadline is not None and self.total_samples[name] else None),
                "window_deadline_miss_fraction": (sum(v > deadline for v in values) / len(values)
                                                  if deadline is not None and values else None),
            }
        return {
            "stages": stages,
            "inference_age_p95_ms": self._percentile(self.inference_age_ms, 0.95),
            "inference_age_total_samples": self.inference_age_total_samples,
            "inference_age_window_samples": len(self.inference_age_ms),
            "inference_age_invalid_samples": self.inference_age_invalid_samples,
            "gpu_peak_mb": round(self.gpu_peak_mb, 2) if self.gpu_peak_mb is not None else None,
            "gpu_device_peak_used_mb": (round(self.gpu_device_peak_used_mb, 2)
                                        if self.gpu_device_peak_used_mb is not None else None),
            "gpu_device_total_mb": (round(self.gpu_device_total_mb, 2)
                                    if self.gpu_device_total_mb is not None else None),
            "gpu_tensor_samples": self.gpu_tensor_samples,
            "gpu_device_samples": self.gpu_device_samples,
            "gpu_sampling_errors": self.gpu_sampling_errors,
            "gpu_last_error": self.gpu_last_error,
            "gpu_memory_units": "MiB; device-wide sampled maximum is not an absolute peak",
        }
