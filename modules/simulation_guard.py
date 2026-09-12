"""Cô lập biến ngoại sinh và bảo vệ lifecycle của CARLA world."""

import atexit
import math
import time


def _sync_settings_match(settings, fps):
    expected_delta = 1.0 / float(fps)
    actual_delta = getattr(settings, "fixed_delta_seconds", None)
    return (bool(getattr(settings, "synchronous_mode", False))
            and actual_delta is not None
            and abs(float(actual_delta) - expected_delta) <= 1e-6)


def enter_synchronous_mode(world, fps, timeout_s=10.0, traffic_manager=None):
    """Enter synchronous mode and recover a partially-applied transition.

    CARLA may apply the setting server-side but time out before acknowledging the
    request. In that state the game thread waits for a tick while the client is
    still waiting for ``apply_settings``. Sending one bounded recovery tick
    breaks the deadlock. The caller receives the original settings for cleanup.
    """
    original = world.get_settings()
    requested = world.get_settings()
    requested.synchronous_mode = True
    requested.fixed_delta_seconds = 1.0 / float(fps)
    recovered = False
    try:
        frame = world.apply_settings(requested, float(timeout_s))
    except RuntimeError as apply_error:
        try:
            frame = world.tick(float(timeout_s))
            if not _sync_settings_match(world.get_settings(), fps):
                raise RuntimeError("world did not retain requested synchronous settings")
            recovered = True
        except Exception as recovery_error:
            try:
                world.apply_settings(original, float(timeout_s))
            except Exception:
                pass
            raise RuntimeError(
                "CARLA synchronous-mode transition failed and recovery tick "
                f"did not restore progress: apply={apply_error}; "
                f"recovery={recovery_error}") from apply_error
    current = world.get_settings()
    if not _sync_settings_match(current, fps):
        try:
            world.apply_settings(original, float(timeout_s))
        except Exception:
            pass
        raise RuntimeError("CARLA acknowledged settings but synchronous mode is inconsistent")
    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(True)
        except Exception:
            try:
                world.apply_settings(original, float(timeout_s))
            except Exception:
                pass
            raise
    return original, frame, recovered


def restore_world_settings(world, original, timeout_s=10.0, traffic_manager=None):
    """Restore world settings, using one tick if the sync exit acknowledgement stalls."""
    tm_error = None
    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception as error:
            tm_error = error
    try:
        frame = world.apply_settings(original, float(timeout_s))
    except RuntimeError as first_error:
        try:
            world.tick(float(timeout_s))
            frame = world.apply_settings(original, float(timeout_s))
        except Exception as retry_error:
            raise RuntimeError(
                "CARLA world settings could not be restored after recovery tick: "
                f"first={first_error}; retry={retry_error}; tm={tm_error}") from first_error
    if tm_error is not None:
        raise RuntimeError(f'Traffic Manager restoration failed: {tm_error}') from tm_error
    return frame


class SynchronousWorldSession:
    """Own one idempotent synchronous-mode session.

    The session can be entered before neural-model warm-up, then have Traffic
    Manager attached later.  This prevents an async/no-VSync CARLA server from
    rendering without a frame cap while CUDA models are being initialized.
    ``close`` is safe to call from both a context manager and a final cleanup.
    """

    def __init__(self, world, fps, timeout_s=10.0, traffic_manager=None):
        self.world = world
        self.fps = int(fps)
        self.timeout_s = float(timeout_s)
        self.traffic_manager = traffic_manager
        self.original_settings = None
        self.entry_frame = None
        self.recovered_transition = False
        self.active = False
        self.traffic_manager_synchronous = False
        self._atexit_registered = False

    def __enter__(self):
        if self.active:
            if (self.traffic_manager is not None
                    and not self.traffic_manager_synchronous):
                self.attach_traffic_manager(self.traffic_manager)
            return self
        (self.original_settings, self.entry_frame,
         self.recovered_transition) = enter_synchronous_mode(
             self.world, self.fps, timeout_s=self.timeout_s,
             traffic_manager=self.traffic_manager)
        self.active = True
        self.traffic_manager_synchronous = self.traffic_manager is not None
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True
        return self

    def attach_traffic_manager(self, traffic_manager):
        """Attach Traffic Manager after the world has already been frozen."""
        if (self.traffic_manager is not None
                and self.traffic_manager is not traffic_manager
                and self.traffic_manager_synchronous):
            raise RuntimeError("cannot replace an active synchronous Traffic Manager")
        self.traffic_manager = traffic_manager
        if self.active and not self.traffic_manager_synchronous:
            try:
                traffic_manager.set_synchronous_mode(True)
                self.traffic_manager_synchronous = True
            except Exception:
                self.close()
                raise
        return traffic_manager

    def close(self):
        if not self.active:
            return None
        original = self.original_settings
        tm = (self.traffic_manager
              if self.traffic_manager_synchronous else None)
        try:
            return restore_world_settings(
                self.world, original, timeout_s=self.timeout_s,
                traffic_manager=tm)
        finally:
            self.abandon()

    def abandon(self):
        """Forget local ownership after failed/unknown cleanup; issue no RPC.

        This prevents an atexit retry against a confirmed dead server. Calling
        it does NOT mean world settings were restored.
        """
        self.active = False
        self.traffic_manager_synchronous = False
        self.original_settings = None
        if self._atexit_registered:
            atexit.unregister(self._atexit_close)
            self._atexit_registered = False

    def _atexit_close(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown must never mask the original failure.
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def map_short_name(name):
    """Normalize ``Carla/Maps/Town02`` and ``Town02`` for safe comparison."""
    return str(name or "").replace("\\", "/").rstrip("/").split("/")[-1].casefold()


def same_carla_map(current_name, requested_name):
    """True when CARLA is already on the requested map.

    Avoiding a redundant ``client.load_world`` matters on constrained Windows
    GPUs: reloading the same Unreal world can transiently double allocations or
    leave the RPC/game thread unavailable between repeated demo runs.
    """
    current = map_short_name(current_name)
    requested = map_short_name(requested_name)
    return bool(current and requested and current == requested)


def _positive_timeout(value, name):
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _probe_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _probe_time(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and nonnegative')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite and nonnegative') from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _probe_map(state, name):
    if not isinstance(state, dict):
        raise ValueError(f'{name} must be a dictionary')
    value = state.get('map')
    if not isinstance(value, str) or not map_short_name(value):
        raise ValueError(f'{name}.map must be explicit')
    return value


def validate_probe_world_selection(before, after, requested_name, reloaded):
    """Validate the fail-closed map/episode/progress contract for a live probe."""
    if not isinstance(requested_name, str) or not map_short_name(requested_name):
        raise ValueError('requested map must be explicit')
    if not isinstance(reloaded, bool):
        raise ValueError('reloaded acknowledgement must be a boolean')

    before_map = _probe_map(before, 'before')
    after_map = _probe_map(after, 'after')
    if not same_carla_map(after_map, requested_name):
        raise RuntimeError(f'expected map {requested_name}, observed {after_map}')

    before_episode = _probe_integer(before.get('episode_id'), 'before.episode_id')
    after_episode = _probe_integer(after.get('episode_id'), 'after.episode_id')
    before_frame = _probe_integer(before.get('frame'), 'before.frame')
    before_time = _probe_time(before.get('simulation_time_s'),
                              'before.simulation_time_s')

    frames = after.get('frames')
    times = after.get('simulation_times')
    if not isinstance(frames, (list, tuple)) or len(frames) != 2:
        raise ValueError('after.frames must contain exactly two frames')
    if not isinstance(times, (list, tuple)) or len(times) != 2:
        raise ValueError('after.simulation_times must contain exactly two times')
    frames = [_probe_integer(value, f'after.frames[{index}]')
              for index, value in enumerate(frames)]
    times = [_probe_time(value, f'after.simulation_times[{index}]')
             for index, value in enumerate(times)]
    if frames[1] <= frames[0] or times[1] <= times[0]:
        raise RuntimeError('post-load map frames/simulation clock did not progress')

    same_map = same_carla_map(before_map, requested_name)
    if same_map:
        if reloaded:
            raise RuntimeError('same-map reuse unexpectedly acknowledged a reload')
        if after_episode != before_episode:
            raise RuntimeError('same-map reuse changed episode')
        if frames[0] < before_frame or times[0] < before_time:
            raise RuntimeError('same-map observation moved backwards')
        mode = 'same-map'
    else:
        if not reloaded:
            raise RuntimeError('map change is missing reload acknowledgement')
        if after_episode == before_episode:
            raise RuntimeError('map change did not create a new episode')
        mode = 'changed-map'

    return {'mode': mode, 'map': after_map, 'episode_id': after_episode,
            'frames': frames, 'simulation_times': times}


def _probe_world_state(world):
    snapshot = world.get_snapshot()
    return {'map': world.get_map().name,
            'episode_id': _probe_integer(world.id, 'world.episode_id'),
            'frame': _probe_integer(snapshot.frame, 'world.frame'),
            'simulation_time_s': _probe_time(
                snapshot.timestamp.elapsed_seconds, 'world.simulation_time_s')}


def observe_loaded_world(client, requested_name, *, timeout_s, client_timeout_s):
    """One read-only observation of an async map; never reload or advance it.

    Each RPC receives the remaining monotonic budget, which is checked again
    after the call. This bounds cooperative CARLA calls, not a native binding
    that ignores its timeout (use an external supervisor for that case).
    Caller supplies its current client timeout for restoration; this client
    must not be shared with another thread during startup observation.
    """
    timeout_s = _positive_timeout(timeout_s, 'timeout_s')
    client_timeout_s = _positive_timeout(client_timeout_s, 'client_timeout_s')
    if not map_short_name(requested_name):
        raise ValueError('requested map must be explicit for observation')
    deadline = time.monotonic() + timeout_s

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('map observation deadline exhausted')
        return left

    def read(call, *, wait=False):
        client.set_timeout(min(client_timeout_s, remaining()))
        result = call(min(client_timeout_s, remaining())) if wait else call()
        remaining()
        return result

    def check_state(world):
        name = read(world.get_map).name
        if not same_carla_map(name, requested_name):
            raise RuntimeError(f'expected map {requested_name}, observed {name}')
        if read(world.get_settings).synchronous_mode:
            raise RuntimeError('map observation found synchronous tick ownership')
        return name

    try:
        world = read(client.get_world)
        episode = world.id
        name = check_state(world)
        before = read(world.get_snapshot)
        after = read(world.wait_for_tick, wait=True)
        times = [float(s.timestamp.elapsed_seconds) for s in (before, after)]
        if (not all(math.isfinite(t) and t >= 0 for t in times)
                or after.frame <= before.frame or times[1] <= times[0]):
            raise RuntimeError('map frames/simulation clock did not progress')
        current = read(client.get_world)
        if current.id != episode:
            raise RuntimeError('map episode changed during observation')
        check_state(current)
        remaining()
        return current, {'map': name, 'episode_id': episode,
                         'frames': [int(before.frame), int(after.frame)],
                         'simulation_times': times}
    finally:
        client.set_timeout(client_timeout_s)


def get_or_load_world(client, requested_name=None, *, recovery_timeout_s=0.0,
                      client_timeout_s=None, diagnostics=None):
    """Return ``(world, reloaded)``; optionally verify a late load completion.

    Recovery is opt-in and observes once, only for a timeout acknowledgement.
    The original load error remains in diagnostics/on the exception chain.
    It never resends the load, changes settings, ticks or restarts the server.
    Legacy callers retain their original exception behavior.
    """
    recovery_timeout_s = float(recovery_timeout_s)
    if not math.isfinite(recovery_timeout_s) or recovery_timeout_s < 0:
        raise ValueError('recovery_timeout_s must be finite and nonnegative')
    if recovery_timeout_s:
        _positive_timeout(client_timeout_s, 'client_timeout_s')
        if not isinstance(diagnostics, dict):
            raise ValueError('recovery requires a diagnostics dictionary')
    world = client.get_world()
    if requested_name and not same_carla_map(
            world.get_map().name, requested_name):
        try:
            return client.load_world(requested_name), True
        except RuntimeError as load_error:
            message = str(load_error)
            if (not recovery_timeout_s
                    or 'timeout' not in message.lower().replace('-', '').replace(' ', '')):
                raise
            diagnostics.update(status='unverified', load_error=message)
            try:
                world, evidence = observe_loaded_world(
                    client, requested_name, timeout_s=recovery_timeout_s,
                    client_timeout_s=client_timeout_s)
            except Exception as observation_error:
                diagnostics['observation_error'] = str(observation_error)
                raise RuntimeError(
                    f'map load timed out; late completion unverified: {observation_error}; '
                    f'original load error: {message}') from load_error
            diagnostics.update(evidence, status='late_completion_verified')
            return world, True
    return world, False


def select_probe_world(client, initial_world, requested_name, *, timeout_s,
                       client_timeout_s):
    """Select and independently observe one map for the probe-only path."""
    before = _probe_world_state(initial_world)
    selected, reloaded = get_or_load_world(client, requested_name)
    selected_map = selected.get_map().name
    selected_episode = _probe_integer(selected.id, 'selected.episode_id')
    if not same_carla_map(selected_map, requested_name):
        raise RuntimeError(f'expected selected map {requested_name}, observed {selected_map}')

    observed, after = observe_loaded_world(
        client, requested_name, timeout_s=timeout_s,
        client_timeout_s=client_timeout_s)
    observed_map = observed.get_map().name
    observed_episode = _probe_integer(observed.id, 'observed.episode_id')
    if (not same_carla_map(observed_map, selected_map)
            or observed_episode != selected_episode):
        raise RuntimeError('selected world changed before probe authorization')

    validated = validate_probe_world_selection(
        before, after, requested_name, reloaded)
    if (validated['episode_id'] != observed_episode
            or not same_carla_map(validated['map'], observed_map)):
        raise RuntimeError('observed world evidence does not match selected world')
    return observed, reloaded, {'before': before, 'after': after,
                                'mode': validated['mode']}


def force_traffic_lights_green(world):
    """Khóa đèn xanh, trả snapshot để caller phục hồi trong ``finally``."""
    import carla

    snapshot = []
    for light in world.get_actors().filter("traffic.traffic_light*"):
        try:
            snapshot.append((light, light.get_state(), light.is_frozen()))
            light.set_state(carla.TrafficLightState.Green)
            light.freeze(True)
        except Exception:
            pass
    return snapshot


def restore_traffic_lights(snapshot):
    for light, state, frozen in snapshot or []:
        try:
            if light.is_alive:
                light.set_state(state)
                light.freeze(frozen)
        except Exception:
            pass
