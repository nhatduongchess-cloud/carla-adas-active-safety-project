import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from modules.pipeline_metrics import PipelineMetrics


class PipelineMetricsTests(unittest.TestCase):
    def test_unmeasured_memory_is_unknown_not_zero(self):
        result = PipelineMetrics().summary()
        for key in ('gpu_peak_mb', 'gpu_device_peak_used_mb', 'gpu_device_total_mb'):
            self.assertIsNone(result[key])

    def test_invalid_ages_are_counted_not_percentiled_or_raised(self):
        metrics = PipelineMetrics()
        for value in (None, -1, float('nan'), float('inf'), 'bad', True):
            self.assertFalse(metrics.record_inference_age(value))
        self.assertTrue(metrics.record_inference_age(12.))
        result = metrics.summary()
        self.assertEqual(result['inference_age_p95_ms'], 12.)
        self.assertEqual(result['inference_age_invalid_samples'], 6)

    def test_gpu_query_failure_retains_tensor_measurement_only(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.max_memory_allocated.return_value = 1024**2
        cuda.mem_get_info.side_effect = RuntimeError('driver unavailable')
        metrics = PipelineMetrics()
        with patch.dict('sys.modules', torch=SimpleNamespace(cuda=cuda)):
            metrics.sample_gpu_memory()
        result = metrics.summary()
        self.assertEqual(result['gpu_peak_mb'], 1.)
        self.assertIsNone(result['gpu_device_peak_used_mb'])
        self.assertEqual(result['gpu_sampling_errors'], 1)

    def test_invalid_gpu_values_never_become_measurements(self):
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.max_memory_allocated.return_value = float('nan')
        cuda.mem_get_info.return_value = (float('nan'), 1024**3)
        metrics = PipelineMetrics()
        with patch.dict('sys.modules', torch=SimpleNamespace(cuda=cuda)):
            metrics.sample_gpu_memory()
        result = metrics.summary()
        self.assertIsNone(result['gpu_peak_mb'])
        self.assertIsNone(result['gpu_device_peak_used_mb'])
        json.dumps(result, allow_nan=False)

    def test_window_and_lifetime_deadline_denominators(self):
        metrics = PipelineMetrics({'safety': 25}, max_samples=2)
        for value in (30, 10, 20):
            metrics.record('safety', value)
        result = metrics.summary()['stages']['safety']
        self.assertEqual(result['deadline_miss_fraction'], 1/3)
        self.assertEqual(result['window_deadline_miss_fraction'], 0.)

    def test_invalid_deadline_rejected(self):
        for value in (-1, float('nan'), float('inf'), 'bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PipelineMetrics({'safety': value})


if __name__ == '__main__':
    unittest.main()
