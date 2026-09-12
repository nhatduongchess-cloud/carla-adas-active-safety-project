"""B04 clock validation at the scheduler and learned-lane safety boundaries."""

import math
import unittest

from modules.inference_scheduler import LatestFrameScheduler
from modules.lane_source import LaneSourceArbiter, map_lane_estimate
from modules.perception_contracts import (
    LaneEstimate, TaggedInferenceResult, validate_frame_age,
)


INVALID_CLOCKS = (
    (float('nan'), 1.0), (float('inf'), 1.0), (float('-inf'), 1.0),
    (1.0, float('nan')), (1.0, float('inf')), (1.0, float('-inf')),
    (1.1, 1.0), (-0.1, 0.0), (0.0, -0.1), (None, 1.0), (1.0, None),
)


def lane(frame_id=10, timestamp=1.0):
    return LaneEstimate(
        frame_id, timestamp, 'learned', confidence=0.9,
        centerline_m=[(0., 0.), (20., 0.)], offset_m=0.,
        lane_width_m=3.5, valid=True,
    )


def completed_scheduler(timestamp=1.0, frame_id=10):
    scheduler = LatestFrameScheduler(lambda payload, _: payload)
    scheduler.submit(frame_id, timestamp, {'detections': ['example']})
    scheduler.close()
    if scheduler.stats()['thread_alive']:
        raise AssertionError('Test worker did not stop')
    return scheduler


class FrameClockTests(unittest.TestCase):
    def test_valid_measurements_keep_age_and_150ms_boundary(self):
        for stamp, now, age in ((0., 0., 0.), (1., 1.125, 125.), (0., .15, 150.)):
            with self.subTest(stamp=stamp, now=now):
                validate_frame_age(10, stamp, 12, now)
                for estimate in (lane(timestamp=stamp),
                                 TaggedInferenceResult(10, stamp, 999., {}, 1.)):
                    self.assertAlmostEqual(estimate.age_ms(now), age)

    def test_age_methods_mark_invalid_clocks_unusable(self):
        for stamp, now in INVALID_CLOCKS:
            for estimate in (lane(timestamp=stamp),
                             TaggedInferenceResult(10, stamp, 999., {}, 1.)):
                with self.subTest(stamp=stamp, now=now, product=type(estimate).__name__):
                    self.assertEqual(estimate.age_ms(now), math.inf)

    def test_validator_rejects_invalid_clocks(self):
        for stamp, now in INVALID_CLOCKS:
            with self.subTest(stamp=stamp, now=now), self.assertRaises(ValueError):
                validate_frame_age(10, stamp, 12, now)

    def test_validator_rejects_invalid_limit(self):
        for limit in (float('nan'), float('inf'), float('-inf'), -1., None):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                validate_frame_age(10, 1., 12, 1., limit)

    def test_future_frames_and_expired_measurements_stay_rejected(self):
        for frame, stamp, now in ((13, 1., 1.), (10, 1., 1.151), (10, 10., 0.)):
            with self.subTest(frame=frame, stamp=stamp), self.assertRaises(ValueError):
                validate_frame_age(frame, stamp, 12, now)


class SchedulerClockTests(unittest.TestCase):
    def test_worker_result_is_admitted_only_during_its_valid_window(self):
        scheduler = completed_scheduler(0.)
        admitted = scheduler.latest(10, .15)
        self.assertIsNotNone(admitted)
        self.assertIs(scheduler.latest(10, .15), admitted)  # cached read is allowed
        self.assertIsNone(scheduler.latest(10, .151))
        self.assertIsNone(scheduler.latest(9, .1))
        self.assertIsNone(scheduler.latest(10, -.1))

    def test_worker_results_with_invalid_clock_do_not_reach_consumers(self):
        for stamp, now in INVALID_CLOCKS:
            if stamp is None or now is None:
                continue  # submit requires numeric timestamps; None means default now
            with self.subTest(stamp=stamp, now=now):
                self.assertIsNone(completed_scheduler(stamp).latest(12, now))

    def test_invalid_max_age_cannot_disable_admission_guard(self):
        scheduler = completed_scheduler()
        for limit in (float('nan'), float('inf'), float('-inf'), -1., None):
            with self.subTest(limit=limit):
                self.assertIsNone(scheduler.latest(12, 1., limit))


class LaneClockTests(unittest.TestCase):
    def test_active_lane_with_invalid_clock_falls_back_to_map(self):
        for stamp, now in INVALID_CLOCKS:
            if now is None:
                continue  # explicitly optional clock gating remains caller-owned
            with self.subTest(stamp=stamp, now=now):
                arbiter = LaneSourceArbiter()
                for frame in range(5):
                    learned = lane(frame)
                    map_lane = map_lane_estimate(frame, 1., [(0., 0.), (20., 0.)])
                    selected = arbiter.select(learned, map_lane, now_s=1., max_age_ms=150.)
                self.assertIs(selected, learned)
                self.assertIs(arbiter.select(lane(5, stamp), map_lane,
                                            now_s=now, max_age_ms=150.), map_lane)


if __name__ == '__main__':
    unittest.main()
