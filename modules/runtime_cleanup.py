"""Runtime teardown coordinator; no driving or perception algorithms.

RPC timeouts and admission budget are cooperative. A native call that ignores
its timeout (or blocked local file I/O) still requires external supervision.
"""
import math
import time


def cleanup_runtime(*, client, world, actors=(), sensor_rig=None,
                    traffic_spawner=None, session=None, traffic_manager=None,
                    original_settings=None, perception=None, telemetry=None,
                    replay=None, journal=None, close_windows=None,
                    primary_error=None, stop_reason=None, budget_s=20.0, rpc_timeout_s=2.0):
    if not all(math.isfinite(v) and v > 0 for v in (budget_s, rpc_timeout_s)):
        raise ValueError('Cleanup budgets must be finite and positive')
    started = time.monotonic()
    deadline = started + budget_s
    steps = {}
    responsive = True

    def attempt(name, action, *, rpc=False):
        nonlocal responsive
        remaining = deadline - time.monotonic()
        if rpc and (not responsive or remaining <= 0):
            steps[name] = {'status': 'UNKNOWN', 'error': 'server unavailable or budget exhausted'}
            return None
        try:
            if rpc:
                client.set_timeout(min(rpc_timeout_s, remaining))
            value = action()
            if value is False:
                raise RuntimeError('operation returned false')
            steps[name] = {'status': 'PASS'}
            return value
        except (Exception, KeyboardInterrupt) as error:
            steps[name] = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}
            if rpc:
                # One bounded liveness check after a failed operation. A local
                # stop error must not skip independent destroys on a live world.
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('cleanup budget exhausted')
                    client.set_timeout(min(1.0, remaining))
                    client.get_world()
                except (Exception, KeyboardInterrupt):
                    responsive = False
            return None

    if perception is not None:
        attempt('neural.stop_measurement', perception.stop_measurement)
        attempt('neural.close', lambda: perception.close(timeout=min(2., max(0., deadline-time.monotonic()))))

        def workers_stopped():
            stats = perception.scheduler_stats()
            return not stats.get('thread_alive', True) and not (stats.get('lane') or {}).get('thread_alive', False)

        attempt('neural.verify', workers_stopped)

    # Probe once initially; do not keep sending RPCs to a confirmed dead server.
    try:
        client.set_timeout(min(1., max(.001, deadline-time.monotonic())))
        current = client.get_world()
        if int(current.id) != int(world.id):
            raise RuntimeError('episode changed before cleanup')
        steps['server.probe'] = {'status': 'PASS'}
    except (Exception, KeyboardInterrupt) as error:
        responsive = False
        steps['server.probe'] = {'status': 'FAIL', 'error': f'{type(error).__name__}: {error}'}

    sensor_actors = list(sensor_rig.actors) if sensor_rig is not None else []
    owned = {int(a.id): a for a in [*sensor_actors,
             *(traffic_spawner.actors if traffic_spawner is not None else ()), *actors]}
    for sensor in sensor_actors:
        attempt(f'sensor.stop.{sensor.id}', lambda s=sensor: s.stop() if s.is_alive else None, rpc=True)
    for actor in reversed(list(owned.values())):
        attempt(f'actor.destroy.{actor.id}', lambda a=actor: a.destroy() if a.is_alive else None, rpc=True)

    if session is not None and session.active:
        # Existing restoration may use up to three RPCs; divide the remaining
        # admission budget rather than giving every retry a fresh total budget.
        previous_timeout = session.timeout_s
        session.timeout_s = max(.001, min(rpc_timeout_s, (deadline-time.monotonic()) / 3))
        try:
            attempt('world.restore', session.close, rpc=True)
        finally:
            session.timeout_s = previous_timeout
            if session.active:
                session.abandon()  # Unknown restoration remains in steps; no atexit RPC retry.
    if traffic_manager is not None:
        attempt('tm.restore', lambda: traffic_manager.set_synchronous_mode(False), rpc=True)

    def verify_world():
        settings = world.get_settings()
        if original_settings is None:
            raise RuntimeError('original world settings unavailable')
        fields = ('synchronous_mode', 'fixed_delta_seconds', 'no_rendering_mode',
                  'substepping', 'max_substeps', 'max_substep_delta_time')
        for name in fields:
            if hasattr(original_settings, name) and getattr(settings, name) != getattr(original_settings, name):
                raise RuntimeError(f'world setting not restored: {name}')
        remaining = set(owned) & {int(a.id) for a in world.get_actors()}
        if remaining:
            raise RuntimeError(f'owned actors remain: {sorted(remaining)}')
        if settings.synchronous_mode:
            raise RuntimeError('cannot certify async progress in a synchronous world')
        before = int(world.get_snapshot().frame)
        timeout = min(rpc_timeout_s, max(.001, deadline-time.monotonic()))
        after = world.wait_for_tick(timeout)
        if after is None or int(after.frame) <= before:
            raise RuntimeError('no post-cleanup asynchronous progress')

    attempt('world.verify', verify_world, rpc=True)
    if journal is not None:
        attempt('journal.exit', lambda: journal.emit('runtime', 'exit',
                                                     stop_reason=stop_reason, error=primary_error))
    for name, owner in (('telemetry', telemetry), ('replay', replay), ('journal', journal)):
        if owner is not None:
            attempt(name + '.close', owner.close)
    if close_windows is not None:
        attempt('display.close', close_windows)
    elapsed = time.monotonic() - started
    if elapsed > budget_s:
        steps['budget'] = {'status': 'FAIL', 'error': 'cooperative cleanup deadline exceeded'}
    return {'verified': all(s['status'] == 'PASS' for s in steps.values()),
            'primary_error': primary_error, 'steps': steps, 'elapsed_s': elapsed,
            'budget_s': budget_s, 'budget_kind': 'cooperative; external native supervisor required'}
