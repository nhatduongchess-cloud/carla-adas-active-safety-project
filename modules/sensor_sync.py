"""Đồng bộ sensor theo đúng CARLA frame và thống kê thời gian chờ."""

from dataclasses import dataclass, field
import queue
import time


class SensorFrameError(RuntimeError):
    """Sensor bỏ mất frame hoặc trả dữ liệu tương lai không thể ghép an toàn."""


@dataclass
class SensorSyncStats:
    waits_ms: list = field(default_factory=list)
    stale_discarded: int = 0
    frame_errors: int = 0
    pairing_skews: list = field(default_factory=list)

    def record(self, wait_ms: float, stale: int) -> None:
        self.waits_ms.append(float(wait_ms))
        self.stale_discarded += int(stale)

    def record_error(self) -> None:
        self.frame_errors += 1

    def record_pairing_skew(self, skew_frames: int) -> None:
        """Camera-vs-LiDAR frame skew for the paired bundle (0 = exact pairing).

        In async-stable mode the camera can lag the LiDAR reference frame (a
        cached RGB reused for several control ticks); a valid at-or-before frame
        raises no ``frame_error``, so this skew is the signal that quantifies the
        temporal misalignment instead of leaving it invisible (B03/B04).
        """
        self.pairing_skews.append(max(0, int(skew_frames)))

    def summary(self) -> dict:
        xs = sorted(self.waits_ms)
        skews = sorted(self.pairing_skews)

        def pct(values, q):
            if not values:
                return 0.0
            idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
            return round(values[idx], 2)

        return {
            "reads": len(xs),
            "wait_p50_ms": pct(xs, 0.50),
            "wait_p95_ms": pct(xs, 0.95),
            "wait_p99_ms": pct(xs, 0.99),
            "stale_discarded": self.stale_discarded,
            "frame_errors": self.frame_errors,
            "pairing_samples": len(skews),
            "pairing_skew_max_frames": (skews[-1] if skews else 0),
            "pairing_skew_p95_frames": pct(skews, 0.95),
        }


def retrieve_exact_frame(sensor_queue, frame_id, timeout=2.0,
                         sensor_name="sensor", stats=None):
    """Bỏ frame cũ nhưng tuyệt đối không ghép frame tương lai với frame yêu cầu."""
    start = time.perf_counter()
    deadline = start + timeout
    stale = 0

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            if stats is not None:
                stats.record_error()
            raise TimeoutError(f"{sensor_name}: timeout chờ CARLA frame {frame_id}")

        try:
            data = sensor_queue.get(timeout=remaining)
        except queue.Empty as exc:
            if stats is not None:
                stats.record_error()
            raise TimeoutError(
                f"{sensor_name}: timeout chờ CARLA frame {frame_id}") from exc
        data_frame = int(data.frame)
        if data_frame < frame_id:
            stale += 1
            continue
        if data_frame > frame_id:
            if stats is not None:
                stats.record_error()
            raise SensorFrameError(
                f"{sensor_name}: thiếu frame {frame_id}, nhận frame tương lai {data_frame}")

        if stats is not None:
            stats.record((time.perf_counter() - start) * 1000.0, stale)
        return data


class OptionalFrameReader:
    """Read an optional sensor without blocking the safety/control deadline.

    A future frame is stashed for the next call instead of being discarded.  A
    timeout or missing frame returns ``None`` and is reflected in sync stats;
    camera/LiDAR continue to use the strict reader above.
    """

    def __init__(self, sensor_queue, sensor_name="optional_sensor", stats=None):
        self.sensor_queue = sensor_queue
        self.sensor_name = sensor_name
        self.stats = stats
        self._pending = None

    def get(self, frame_id, timeout=0.005):
        start = time.perf_counter()
        deadline = start + max(0.0, float(timeout))
        stale = 0
        while True:
            data = self._pending
            self._pending = None
            if data is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    if self.stats is not None:
                        self.stats.record_error()
                        self.stats.record((time.perf_counter() - start) * 1000.0, stale)
                    return None
                try:
                    data = self.sensor_queue.get(timeout=remaining)
                except queue.Empty:
                    if self.stats is not None:
                        self.stats.record_error()
                        self.stats.record((time.perf_counter() - start) * 1000.0, stale)
                    return None

            data_frame = int(data.frame)
            if data_frame < int(frame_id):
                stale += 1
                continue
            if data_frame > int(frame_id):
                self._pending = data
                if self.stats is not None:
                    self.stats.record_error()
                    self.stats.record((time.perf_counter() - start) * 1000.0, stale)
                return None
            if self.stats is not None:
                self.stats.record((time.perf_counter() - start) * 1000.0, stale)
            return data


def warmup_sensor_streams(world, sensor_queues, attempts=20, timeout=0.5,
                          tick_timeout=None):
    """Bootstrap sensor GPU trong synchronous mode mà không deadlock tick đầu.

    ``sensor_queues`` là mapping ``name -> queue``. Chỉ trả về khi mọi sensor đã
    cung cấp đúng cùng frame; lỗi warm-up không tính vào KPI runtime.
    """
    last_error = None
    for _ in range(max(1, int(attempts))):
        frame_id = (world.tick() if tick_timeout is None
                    else world.tick(float(tick_timeout)))
        ready = True
        for name, sensor_queue in sensor_queues.items():
            try:
                retrieve_exact_frame(
                    sensor_queue, frame_id, timeout=timeout, sensor_name=name)
            except (TimeoutError, SensorFrameError) as exc:
                last_error = exc
                ready = False
                break
        if ready:
            return frame_id
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(
        f"sensor warm-up thất bại sau {attempts} tick{detail}")


def put_latest(sensor_queue, data):
    """Non-blocking sensor callback that bounds latency and memory."""
    try:
        sensor_queue.put_nowait(data)
        return False
    except queue.Full:
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            pass
        sensor_queue.put_nowait(data)
        return True


class LatestFrameReader:
    """Read the newest async sensor sample without accepting future frames."""

    def __init__(self, sensor_queue, sensor_name="sensor", stats=None,
                 pending_capacity=8):
        self.sensor_queue = sensor_queue
        self.sensor_name = sensor_name
        self.stats = stats
        self.pending_capacity = max(1, int(pending_capacity))
        self._pending = []

    def _collect(self, timeout):
        items = list(self._pending)
        self._pending = []
        if not items and timeout > 0.0:
            try:
                items.append(self.sensor_queue.get(timeout=timeout))
            except queue.Empty:
                return items
        while True:
            try:
                items.append(self.sensor_queue.get_nowait())
            except queue.Empty:
                return items

    def get_latest(self, timeout=0.0):
        start = time.perf_counter()
        items = self._collect(max(0.0, float(timeout)))
        if not items:
            return None
        result = max(items, key=lambda item: int(item.frame))
        if self.stats is not None:
            self.stats.record(
                (time.perf_counter() - start) * 1000.0,
                max(0, len(items) - 1))
        return result

    def get_at_or_before(self, frame_id, timeout=0.0):
        start = time.perf_counter()
        items = self._collect(max(0.0, float(timeout)))
        eligible = [item for item in items if int(item.frame) <= int(frame_id)]
        future = sorted(
            (item for item in items if int(item.frame) > int(frame_id)),
            key=lambda item: int(item.frame))
        self._pending = future[-self.pending_capacity:]
        if not eligible:
            return None
        result = max(eligible, key=lambda item: int(item.frame))
        if self.stats is not None:
            self.stats.record(
                (time.perf_counter() - start) * 1000.0,
                max(0, len(eligible) - 1))
        return result
