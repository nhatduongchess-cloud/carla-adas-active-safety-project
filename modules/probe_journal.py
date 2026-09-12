"""Durable evidence for bounded, disposable CARLA diagnostic workers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import queue
import threading
import time


class ProbeJournal:
    def __init__(self, stream):
        self.stream = stream
        self.started = time.monotonic()

    def emit(self, stage, event, **fields):
        row = {'utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_s': round(time.monotonic() - self.started, 4),
               'stage': stage, 'event': event, **fields}
        self.stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.stream.flush()  # preserve the last API entered if native code dies

    @contextmanager
    def stage(self, name, **fields):
        self.emit(name, 'begin', **fields)
        try:
            yield
        except BaseException as exc:
            self.emit(name, 'error', error=f'{type(exc).__name__}: {exc}')
            raise
        else:
            self.emit(name, 'end')

    def call(self, name, function, *args, **kwargs):
        with self.stage(name):
            return function(*args, **kwargs)


def _diagnostic_value(value, depth=0):
    """Bound input size; never serialize CARLA actors or arbitrary repr methods."""
    if depth > 5:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, dict):
        return {k[:80]: _diagnostic_value(v, depth + 1)
                for k, v in list(value.items())[:32] if isinstance(k, str)}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(v, depth + 1) for v in value[:32]]
    return None


class RuntimeCrashJournal:
    """Opt-in prefix journal, with no file I/O on the control-loop producer.

    One daemon writer reuses ProbeJournal (flush per record). Full queue/cap or
    writer failure drops diagnostics, never control. Snapshot getters are
    sampled on the caller; no extra tick/wait/set operation is performed.
    Shutdown is bounded, so a blocked writer/native process exit can lose the
    queued tail. This is diagnostic evidence, not a safety/performance gate.
    """

    def __init__(self, path=None, *, stream=None, max_records=12000,
                 max_bytes=8 * 1024 * 1024, queue_capacity=256):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (max_records, max_bytes, queue_capacity)):
            raise ValueError('Journal limits must be positive integers')
        if path is not None and stream is not None:
            raise ValueError('Select path or injected stream, not both')
        self.enabled = path is not None or stream is not None
        self.path = str(path) if path is not None else None
        self.max_records, self.max_bytes = max_records, max_bytes
        self.accepted = self.written = self.dropped = self.bytes_written = 0
        self.capacity_reached = False
        self._byte_cap_reached = False
        self.writer_error = None
        self._closed = threading.Event()
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._thread = None
        self._stream = stream
        self._owns_stream = path is not None
        if self.enabled:
            # Explicit startup opt-in: refuse overwrite and do not create parents.
            if path is not None:
                self._stream = open(path, 'x', encoding='utf-8', newline='\n')
            self._thread = threading.Thread(target=self._write_loop,
                                            name='runtime-crash-journal', daemon=True)
            try:
                self._thread.start()
            except Exception:
                if self._owns_stream:
                    self._stream.close()
                raise

    def emit(self, stage, event, **fields):
        if not self.enabled:
            return
        if (self._closed.is_set() or self.writer_error or self.capacity_reached
                or self.accepted >= self.max_records):
            self.dropped += 1
            if self.accepted >= self.max_records:
                self.capacity_reached = True
            return
        try:
            fields = _diagnostic_value(fields)
            fields['sample_monotonic_s'] = time.monotonic()
            fields['sample_utc'] = datetime.now(timezone.utc).isoformat()
            fields['journal_counts_at_enqueue'] = {
                'accepted': self.accepted, 'dropped': self.dropped, 'written': self.written}
            self._queue.put_nowait((str(stage)[:80], str(event)[:80], fields))
            self.accepted += 1
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.dropped += 1

    def _write_loop(self):
        owner = self

        class LimitedStream:
            def write(self, value):
                count = len(value.encode('utf-8'))
                if owner.bytes_written + count > owner.max_bytes:
                    owner.capacity_reached = True
                    owner._byte_cap_reached = True
                    return
                owner._stream.write(value)
                owner.bytes_written += count

            def flush(self):
                owner._stream.flush()

        journal = ProbeJournal(LimitedStream())
        try:
            while not self._closed.is_set() or not self._queue.empty():
                try:
                    stage, event, fields = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                before = self.bytes_written
                if self._byte_cap_reached:
                    self.dropped += 1
                    continue
                journal.emit(stage, event, **fields)
                if self.bytes_written > before:
                    self.written += 1
                else:
                    self.dropped += 1
        except Exception as error:
            # Do not print to potentially broken stdout or retry failing disk I/O.
            self.writer_error = type(error).__name__
            self.dropped += 1
        finally:
            if self._owns_stream:
                try:
                    self._stream.close()
                except Exception as error:
                    self.writer_error = self.writer_error or type(error).__name__

    def record_runtime(self, phase, *, world, ego, carla_map=None,
                       read_control=False, **fields):
        if not self.enabled:
            return
        if (self._closed.is_set() or self.writer_error or self.capacity_reached
                or self._queue.full()):
            self.dropped += 1
            return
        data = dict(snapshot_frame=None, simulation_time_s=None, pose=None,
                    velocity_mps=None, road=None, control_readback=None,
                    control_readback_semantics='last cached vehicle control; may lag submission',
                    unavailable=[])
        try:
            snapshot = world.get_snapshot()
            data['snapshot_frame'] = snapshot.frame
            data['simulation_time_s'] = snapshot.timestamp.elapsed_seconds
            actor = snapshot.find(ego.id)
            if actor is None:
                data['unavailable'].append('actor_snapshot')
            else:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                data['pose'] = {
                    'location': {axis: getattr(transform.location, axis) for axis in ('x','y','z')},
                    'rotation_deg': {axis: getattr(transform.rotation, axis) for axis in ('pitch','yaw','roll')}}
                data['velocity_mps'] = {axis: getattr(velocity, axis) for axis in ('x','y','z')}
                if carla_map is not None:
                    try:
                        wp = carla_map.get_waypoint(transform.location, project_to_road=True)
                        if wp is None:
                            data['unavailable'].append('waypoint')
                        else:
                            data['road'] = {key: getattr(wp, key) for key in ('road_id','lane_id','is_junction')}
                    except Exception:
                        data['unavailable'].append('waypoint')
                else:
                    data['unavailable'].append('map')
        except Exception:
            data['unavailable'].append('snapshot')
        if read_control:
            try:
                control = ego.get_control()
                data['control_readback'] = {key: getattr(control, key)
                                            for key in ('throttle','brake','steer','hand_brake')}
            except Exception:
                data['unavailable'].append('control_readback')
        data.update(fields)
        self.emit(phase, 'sample', **data)

    def summary(self):
        return {'enabled': self.enabled, 'path': self.path, 'accepted': self.accepted,
                'written': self.written, 'dropped': self.dropped,
                'pending': self._queue.qsize(), 'bytes_written': self.bytes_written,
                'max_records': self.max_records, 'max_bytes': self.max_bytes,
                'capacity_reached': self.capacity_reached, 'writer_error': self.writer_error,
                'writer_alive': bool(self._thread and self._thread.is_alive())}

    def close(self, timeout=0.5):
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0., min(float(timeout), 1.)))


def read_events(path):
    """A worker may die in the middle of its final JSON line."""
    events = []
    if path.exists():
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    events.append(row)
            except json.JSONDecodeError:
                continue
    return events


def validate_capture_events(events, required_sensors, requested_frames):
    """Validate one exact, contiguous synchronous multisensor capture."""
    if (not isinstance(requested_frames, int) or isinstance(requested_frames, bool)
            or requested_frames < 1):
        raise ValueError('requested_frames must be a positive integer')
    if (not isinstance(required_sensors, (list, tuple)) or not required_sensors
            or any(not isinstance(name, str) or not name for name in required_sensors)
            or len(set(required_sensors)) != len(required_sensors)):
        raise ValueError('required_sensors must be a non-empty unique sequence')

    sensors = list(required_sensors)
    rows = [row for row in events if isinstance(row, dict)]
    errors = []
    map_rows = [row for row in rows
                if row.get('stage') == 'map' and row.get('event') == 'info']
    if len(map_rows) != 1:
        errors.append(f'expected one selected map event, found {len(map_rows)}')
    completions = [(index, row) for index, row in enumerate(rows)
                   if row.get('stage') == 'capture' and row.get('event') == 'complete']
    frame_rows = [(index, row) for index, row in enumerate(rows)
                  if row.get('event') == 'frame'
                  and isinstance(row.get('stage'), str)
                  and row['stage'].startswith('capture.')]
    allowed_stages = {'capture.tick', *(f'capture.{name}' for name in sensors)}
    unknown = [row.get('stage') for _, row in frame_rows
               if row.get('stage') not in allowed_stages]
    if unknown:
        errors.append(f'unknown capture frame stages: {unknown}')
    if len(completions) != 1:
        errors.append(f'expected one capture.complete, found {len(completions)}')

    expected_stages = []
    for _ in range(requested_frames):
        expected_stages.extend(['capture.tick', *(f'capture.{name}' for name in sensors)])
    actual_stages = [row.get('stage') for _, row in frame_rows]
    if actual_stages != expected_stages:
        errors.append('capture frame count or order does not match request')

    ticks = [row for _, row in frame_rows if row.get('stage') == 'capture.tick']
    sensor_rows = {name: [row for _, row in frame_rows
                          if row.get('stage') == f'capture.{name}']
                   for name in sensors}
    frames = [row.get('world_frame') for row in ticks]
    if (len(ticks) != requested_frames
            or any(not isinstance(frame, int) or isinstance(frame, bool) for frame in frames)):
        errors.append('tick frames must be exact integers for every requested frame')
    elif any(current != previous + 1 for previous, current in zip(frames, frames[1:])):
        errors.append('tick world frames are not contiguous increments of one')

    tick_indices = [row.get('index') for row in ticks]
    if tick_indices != list(range(requested_frames)):
        errors.append('tick indices must be exactly 0..requested_frames-1')

    for name, values in sensor_rows.items():
        if len(values) != requested_frames:
            errors.append(f'{name} frame count does not match request')
            continue
        indices = [row.get('index') for row in values]
        if indices != list(range(requested_frames)):
            errors.append(f'{name} indices must be exactly 0..requested_frames-1')
        for index, row in enumerate(values):
            expected = frames[index] if index < len(frames) else None
            world_frame, sensor_frame = row.get('world_frame'), row.get('sensor_frame')
            if (not isinstance(world_frame, int) or isinstance(world_frame, bool)
                    or not isinstance(sensor_frame, int) or isinstance(sensor_frame, bool)
                    or expected is None or world_frame != expected or sensor_frame != expected):
                errors.append(f'{name} frame {index} is not aligned to its world tick')

    capture_rows = [*map_rows, *(row for _, row in frame_rows)]
    if len(completions) == 1:
        complete_index, complete = completions[0]
        if any(index > complete_index for index, _ in frame_rows):
            errors.append('capture frames occur after capture.complete')
        if complete.get('frames') != requested_frames:
            errors.append('capture.complete frame count does not match request')
        if complete.get('required_sensors') != sensors:
            errors.append('capture.complete sensors do not match request')
        capture_rows.append(complete)

    episodes = [row.get('episode_id') for row in capture_rows]
    if (not episodes or any(not isinstance(value, int) or isinstance(value, bool)
                            for value in episodes) or len(set(episodes)) != 1):
        errors.append('capture rows must share one integer episode_id')
        episode_id = None
    else:
        episode_id = episodes[0]

    counts = {'ticks': len(ticks), 'complete': len(completions),
              'sensors': {name: len(values) for name, values in sensor_rows.items()}}
    return {'valid': not errors, 'status': 'PASS' if not errors else 'FAIL',
            'errors': errors, 'frames': frames,
            'episode_id': episode_id, 'counts': counts, 'index_source': 'explicit'}


def summarize_probe(events, returncode, timed_out=False, *,
                    required_sensors=None, requested_frames=None):
    # Never call a capture PASS when cleanup failed or the process was killed.
    if (required_sensors is None) != (requested_frames is None):
        raise ValueError('required_sensors and requested_frames must be provided together')
    capture_integrity = ({'valid': None, 'status': 'NOT_REQUESTED'}
                         if required_sensors is None else
                         validate_capture_events(events, required_sensors, requested_frames))
    completed = any(e.get('stage') == 'result' and e.get('event') == 'pass' for e in events)
    cleaned = any(e.get('stage') == 'cleanup' and e.get('event') == 'verified' for e in events)
    errors = [e for e in events if e.get('event') == 'error']
    capture_valid = required_sensors is None or capture_integrity['valid']
    passed = (returncode == 0 and not timed_out and completed and cleaned
              and not errors and capture_valid)
    pending = []
    for row in events:
        key = row.get('stage')
        if row.get('event') == 'begin':
            pending.append(key)
        elif row.get('event') in ('end', 'error') and key in pending:
            pending.remove(key)
    return {'status': 'PASS' if passed else 'FAIL', 'returncode': returncode,
            'timed_out': timed_out, 'cleanup_verified': cleaned,
            'requires_world_check': not cleaned, 'unfinished_stages': pending,
            'errors': errors, 'capture_integrity': capture_integrity,
            'capture_verified': capture_integrity['valid'],
            'capture_errors': capture_integrity.get('errors', []),
            'last_event': events[-1] if events else None}
