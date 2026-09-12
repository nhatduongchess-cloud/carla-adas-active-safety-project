"""Isolate CARLA startup one component at a time; never launch/restart server.

The parent keeps a wall-clock deadline even if a native CARLA call hangs/crashes.
It kills only its own worker on timeout, never the server or other clients.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from modules.probe_journal import ProbeJournal, read_events, summarize_probe

SENSORS = ('rgb', 'semantic', 'depth', 'lidar', 'radar')


def sensor_names(value):
    names = value.split(',')
    if (not names or names[0] != 'rgb' or len(set(names)) != len(names)
            or any(name not in SENSORS for name in names)):
        raise argparse.ArgumentTypeError('Use unique sensor names starting with rgb: ' + ','.join(SENSORS))
    return names


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ego_state_snapshot(ego, carla_map):
    """Return JSON-safe ego state for crash isolation without changing control."""
    location = ego.get_location()
    velocity = ego.get_velocity()
    transform = ego.get_transform()
    position = {axis: _finite_number(getattr(location, axis, None))
                for axis in ('x', 'y', 'z')}
    velocity_mps = {axis: _finite_number(getattr(velocity, axis, None))
                    for axis in ('x', 'y', 'z')}
    speed_mps = math.sqrt(sum((component or 0.0) ** 2
                              for component in velocity_mps.values()))
    state = {
        'position_m': position,
        'velocity_mps': velocity_mps,
        'speed_mps': round(speed_mps, 4),
        'rotation_deg': {axis: _finite_number(getattr(transform.rotation, axis, None))
                         for axis in ('pitch', 'yaw', 'roll')},
        'road_id': None,
        'section_id': None,
        'lane_id': None,
        'is_junction': None,
    }
    try:
        waypoint = carla_map.get_waypoint(location, project_to_road=True)
        if waypoint is not None:
            for name in ('road_id', 'section_id', 'lane_id', 'is_junction'):
                value = getattr(waypoint, name, None)
                state[name] = bool(value) if name == 'is_junction' else value
    except Exception as exc:
        state['waypoint_error'] = f'{type(exc).__name__}: {exc}'
    return state


def crash_snapshot():
    root = Path(os.environ.get('LOCALAPPDATA', '')) / 'CarlaUE4/Saved/Crashes'
    try:
        return {str(p): p.stat().st_mtime_ns for p in root.glob('*/CrashContext.runtime-xml')}
    except OSError:
        return {}


def new_crash_reports(before, after):
    reports = []
    for path, modified in after.items():
        if before.get(path) == modified:
            continue
        entry = {'path': path, 'modified_ns': modified}
        try:
            root = ET.parse(path).getroot()
            for tag in ('CrashType', 'ErrorMessage', 'SecondsSinceStart', 'ProcessId'):
                entry[tag] = root.findtext('.//' + tag)
        except (OSError, ET.ParseError) as exc:
            entry['read_error'] = str(exc)
        reports.append(entry)
    return reports


def managed_host_check(manifest_path, *, host, port, expected_digest=None,
                       nonce=None):
    """Validate one managed server without importing or calling CARLA."""
    from modules.carla_host_guard import (
        allowed_python_chain, load_managed_manifest, query_windows_host,
        validate_managed_server, verify_claim,
    )

    if (expected_digest is None) != (nonce is None):
        raise ValueError('Managed worker digest and claim nonce must be supplied together')
    manifest, digest = load_managed_manifest(manifest_path)
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError('Managed-server manifest digest changed')
    endpoint = manifest['endpoint']
    if host != endpoint['host'] or port != endpoint['ports'][0]:
        raise RuntimeError('Probe endpoint does not match managed-server manifest')
    snapshot = query_windows_host()
    allowed = allowed_python_chain(snapshot)
    if nonce is not None:
        marker = verify_claim(manifest_path, digest, manifest['run_id'], nonce)
        if marker['supervisor_pid'] not in allowed:
            raise RuntimeError('Managed worker is not a child of the claiming supervisor')
    root = Path(__file__).resolve().parent
    evidence = validate_managed_server(
        manifest, snapshot, expected_install_root=root.parent,
        project_root=root, allowed_python_pids=allowed)
    return {'manifest': manifest, 'manifest_sha256': digest, 'host': evidence}


class WorkerResult(tuple):
    """A three-item legacy tuple with explicit termination evidence."""

    def __new__(cls, returncode, timed_out, interrupted, termination):
        result = super().__new__(cls, (returncode, timed_out, interrupted))
        result.termination = termination
        return result


def supervise_worker(command, log, budget):
    """Bound only the owned worker; return even after Ctrl+C with a failure flag."""
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, shell=False)
    timed_out, interrupted = False, False
    termination = {'forced': False, 'reason': None, 'kill_attempted': False,
                   'kill_succeeded': None, 'exit_verified': False, 'error': None}
    try:
        proc.wait(timeout=budget)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        timed_out = isinstance(exc, subprocess.TimeoutExpired)
        interrupted = isinstance(exc, KeyboardInterrupt)
        termination.update(forced=True,
                           reason='timeout' if timed_out else 'keyboard_interrupt')
        try:
            if proc.poll() is None:
                termination['kill_attempted'] = True
                proc.kill()
                termination['kill_succeeded'] = True
        except Exception as error:
            termination['kill_succeeded'] = False
            termination['error'] = f'{type(error).__name__}: {error}'
        try:
            proc.wait(timeout=10)
        except Exception as error:
            detail = f'{type(error).__name__}: {error}'
            termination['error'] = '; '.join(
                value for value in (termination['error'], detail) if value)
    try:
        termination['exit_verified'] = proc.poll() is not None
    except Exception as error:
        termination['error'] = termination['error'] or f'{type(error).__name__}: {error}'
    return WorkerResult(proc.returncode, timed_out, interrupted, termination)


class OwnedWorld:
    """Track actors immediately, even when a reused spawn helper later raises."""
    def __init__(self, world, actors, journal):
        self.world, self.actors, self.journal = world, actors, journal

    def __getattr__(self, name):
        return getattr(self.world, name)

    def try_spawn_actor(self, blueprint, transform):
        actor = self.journal.call('spawn.' + blueprint.id, self.world.try_spawn_actor,
                                  blueprint, transform)
        if actor is not None:
            self.actors.append(actor)
            self.journal.emit('actors', 'owned', id=int(actor.id), type_id=actor.type_id,
                              owned_count=len(self.actors))
        return actor


def worker(args, journal):
    # No CARLA, Torch, OpenCV or model loading in the supervisor/offline tests.
    import queue
    import random
    from types import SimpleNamespace
    import carla
    import config as cfg
    from smoke_carla_camera import spawn_test_vehicle
    from collect_carla_dataset import spawn_heavy_vehicles, spawn_two_wheelers, spawn_walkers
    from modules.sensor_setup import (configure_camera_blueprint, configure_lidar_blueprint,
                                      configure_radar_blueprint)
    from modules.sensor_sync import retrieve_exact_frame, put_latest, SensorSyncStats
    from modules.simulation_guard import (get_or_load_world, select_probe_world,
                                          SynchronousWorldSession)

    actors, sensors, queues = [], [], {}
    world, session = None, None
    success, cleaned = False, True
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    try:
        version = journal.call('server.version', client.get_server_version)
        journal.emit('version', 'info', server=version, client=client.get_client_version())
        if version != client.get_client_version():
            raise RuntimeError('Client/server version mismatch; refusing simulation mutation')
        current = journal.call('world.get', client.get_world)
        if current.get_settings().synchronous_mode:
            raise RuntimeError('World already synchronous; another/stale client may own ticks')
        existing = current.get_actors()
        busy = [int(a.id) for a in existing if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.'))]
        if busy:
            raise RuntimeError(f'World has user/dynamic actors; refusing probe: {busy[:10]}')
        if args.town:
            world, reloaded, transition = journal.call(
                'world.map', select_probe_world, client, current, args.town,
                timeout_s=args.timeout, client_timeout_s=args.timeout)
        else:
            world, reloaded = journal.call('world.map', get_or_load_world,
                                           client, args.town)
            transition = None
        episode_id = int(world.id)
        journal.emit('map', 'info', name=world.get_map().name,
                     reloaded=reloaded, episode_id=episode_id,
                     transition=transition)
        carla_map = world.get_map()
        session = SynchronousWorldSession(world, args.fps, timeout_s=args.timeout)
        journal.call('sync.enter', session.__enter__)
        owner = OwnedWorld(world, actors, journal)
        ego = journal.call('ego.spawn', spawn_test_vehicle, owner)
        journal.call('ego.stabilize', world.tick, args.timeout)
        library = world.get_blueprint_library()
        probe_cfg = SimpleNamespace(**{name: getattr(cfg, name) for name in dir(cfg) if name.isupper()})
        probe_cfg.CAM_WIDTH, probe_cfg.CAM_HEIGHT = args.width, args.height
        probe_cfg.CAMERA_POSTPROCESS = args.postprocess
        probe_cfg.FPS = args.fps
        probe_cfg.CAMERA_SENSOR_TICK_S = 0.0
        probe_cfg.LIDAR_SENSOR_TICK_S = probe_cfg.RADAR_SENSOR_TICK_S = 0.0

        def spawn_extras():
            tm = None
            tm_mode = getattr(args, 'tm_mode', 'autopilot')
            journal.emit('tm.mode', 'info',
                         mode=tm_mode if args.traffic_manager else 'disabled',
                         ego_autopilot=bool(args.traffic_manager and tm_mode == 'autopilot'))
            if args.traffic_manager:
                tm = journal.call('tm.get', client.get_trafficmanager, cfg.TM_PORT)
                journal.call('tm.attach', session.attach_traffic_manager, tm)
                journal.call('tm.seed', tm.set_random_device_seed, args.seed)
                if tm_mode == 'autopilot':
                    journal.call('ego.autopilot', ego.set_autopilot, True, tm.get_port())
            rng = random.Random(args.seed)
            for name, count, spawn in (('heavy', args.heavy_vehicles, spawn_heavy_vehicles),
                                       ('two_wheelers', args.two_wheelers, spawn_two_wheelers)):
                if count:
                    spawned = journal.call(name + '.setup', spawn, owner, tm, count, rng,
                                           reference_vehicle=ego)
                    if len(spawned) != count:
                        raise RuntimeError(f'{name}: requested {count}, spawned {len(spawned)}')
            if args.walkers:
                walkers, _ = journal.call('walkers.setup', spawn_walkers, owner, args.walkers,
                                           rng, reference_vehicle=ego)
                if len(walkers) != args.walkers:
                    raise RuntimeError(f'walkers: requested {args.walkers}, spawned {len(walkers)}')

        def spawn_sensors():
            for name in args.sensors:
                if name in ('rgb', 'semantic', 'depth'):
                    suffix = 'semantic_segmentation' if name == 'semantic' else name
                    bp = configure_camera_blueprint(library.find('sensor.camera.' + suffix), probe_cfg)
                    location = (cfg.CAM_X, cfg.CAM_Y, cfg.CAM_Z)
                elif name == 'lidar':
                    bp = configure_lidar_blueprint(library.find('sensor.lidar.ray_cast'), probe_cfg)
                    location = (cfg.LIDAR_X, cfg.LIDAR_Y, cfg.LIDAR_Z)
                else:
                    bp = configure_radar_blueprint(library.find('sensor.other.radar'), probe_cfg)
                    location = (cfg.RADAR_X, cfg.RADAR_Y, cfg.RADAR_Z)
                sensor = journal.call('sensor.' + name + '.spawn', world.spawn_actor, bp,
                                      carla.Transform(carla.Location(*location)), attach_to=ego)
                actors.append(sensor); sensors.append(sensor)
                q = queue.Queue(maxsize=8); queues[name] = q
                journal.emit('sensor.' + name, 'owned', id=int(sensor.id), owned_count=len(actors))
                journal.call('sensor.' + name + '.listen', sensor.listen,
                             lambda data, target=q: put_latest(target, data))

        if args.spawn_order == 'actors-first':
            spawn_extras(); spawn_sensors()
        else:
            spawn_sensors(); spawn_extras()
        # Bounded warmup with explicit tick deadlines. Radar is required here
        # only to validate its sensor, NOT to change runtime's optional policy.
        ready = False
        for attempt in range(10):
            journal.emit('ego.state', 'sample', phase='warmup', attempt=attempt,
                         requested_motion=args.ego_motion,
                         **ego_state_snapshot(ego, carla_map))
            frame = journal.call('warmup.tick', world.tick, args.timeout)
            try:
                for name, q in queues.items():
                    retrieve_exact_frame(q, frame, timeout=0.5, sensor_name=name)
                ready = True
                break
            except (TimeoutError, RuntimeError) as exc:
                journal.emit('warmup', 'retry', attempt=attempt, detail=str(exc))
        if not ready:
            raise TimeoutError('Sensors did not produce an aligned warmup frame')
        stats = {name: SensorSyncStats() for name in queues}
        for capture_index in range(args.frames):
            journal.emit('ego.state', 'sample', phase='capture', index=capture_index,
                         requested_motion=args.ego_motion,
                         **ego_state_snapshot(ego, carla_map))
            if args.ego_motion == 'manual-forward':
                control = carla.VehicleControl(throttle=args.manual_throttle,
                                               steer=0.0, brake=0.0,
                                               hand_brake=False)
                journal.call('ego.manual_control', ego.apply_control, control)
            frame = journal.call('capture.tick', world.tick, args.timeout)
            journal.emit('capture.tick', 'frame', index=capture_index,
                         world_frame=int(frame), episode_id=episode_id)
            with journal.stage('capture.read', frame=frame):
                for name, q in queues.items():
                    data = retrieve_exact_frame(q, frame, timeout=args.timeout,
                                                sensor_name=name, stats=stats[name])
                    journal.emit('capture.' + name, 'frame', index=capture_index,
                                 world_frame=int(frame), sensor_frame=int(data.frame),
                                 episode_id=episode_id, bytes=len(data.raw_data))
        journal.emit('capture', 'complete', frames=args.frames,
                     required_sensors=list(args.sensors), episode_id=episode_id,
                     sensor_stats={n: s.summary() for n, s in stats.items()})
        success = True
    except Exception as exc:
        journal.emit('worker', 'error', error=f'{type(exc).__name__}: {exc}')
    finally:
        # Probe liveness before teardown.  Once the native server is gone,
        # sensor.stop()/actor.destroy() can each block for the full CARLA RPC
        # timeout.  Skip those calls and retain cleanup_verified=false so a
        # crash is never reported as a clean PASS.
        probe_timeout = min(max(float(args.timeout), 0.1), 1.0)
        cleanup_rpc_timeout = min(max(float(args.timeout), 0.1), 5.0) if not success else args.timeout

        def probe_server(restore_timeout):
            reachable = True
            try:
                client.set_timeout(probe_timeout)
                client.get_world()
                journal.emit('cleanup.server_probe', 'reachable', timeout_s=probe_timeout)
            except Exception as exc:
                reachable = False
                journal.emit('cleanup.server_probe', 'unavailable',
                             timeout_s=probe_timeout,
                             error=f'{type(exc).__name__}: {exc}')
            finally:
                try:
                    client.set_timeout(restore_timeout)
                except Exception as exc:
                    reachable = False
                    journal.emit('cleanup.server_probe', 'timeout_restore_error',
                                 error=f'{type(exc).__name__}: {exc}')
            return reachable

        server_reachable = probe_server(args.timeout)
        if not server_reachable:
            cleaned = False

        if not server_reachable:
            journal.emit('cleanup', 'skipped', reason='server_unavailable',
                         owned_actor_count=len(actors))
        else:
            try:
                client.set_timeout(cleanup_rpc_timeout)
            except Exception as exc:
                server_reachable = False
                cleaned = False
                journal.emit('cleanup', 'error',
                             error=f'Unable to set cleanup timeout: {type(exc).__name__}: {exc}')

            # Each cleanup step is independent: camera.stop must not prevent restore.
            for sensor in reversed(sensors):
                if not server_reachable:
                    break
                try:
                    journal.call(f'cleanup.stop.{sensor.id}', sensor.stop)
                except Exception:
                    cleaned = False
                    # A teardown RPC failure can race with native server exit;
                    # do not issue another potentially blocking RPC.
                    server_reachable = False
            for actor in reversed(actors):
                if not server_reachable:
                    break
                try:
                    destroyed = journal.call(f'cleanup.destroy.{actor.id}', actor.destroy)
                    if not destroyed:
                        cleaned = False
                        journal.emit('cleanup', 'error', error=f'Actor {actor.id} destroy returned false')
                        # Treat a false destroy as a transport/lifecycle
                        # failure and abort teardown before session.restore.
                        server_reachable = False
                except Exception:
                    cleaned = False
                    # Do not retry a possibly dead native server.
                    server_reachable = False
            if server_reachable and session is not None:
                try:
                    client.set_timeout(args.timeout)
                    journal.call('cleanup.restore', session.close)
                    snapshot = journal.call('cleanup.async_tick', world.wait_for_tick, args.timeout)
                    remaining = {int(a.id) for a in world.get_actors()} & {int(a.id) for a in actors}
                    if world.get_settings().synchronous_mode or remaining:
                        raise RuntimeError(f'Cleanup incomplete: surviving owned actors {sorted(remaining)}')
                    journal.emit('cleanup', 'verified', frame=int(snapshot.frame)) if cleaned else None
                except Exception as exc:
                    journal.emit('cleanup', 'error', error=str(exc))
                    cleaned = False
            if not server_reachable:
                journal.emit('cleanup', 'skipped', reason='server_unavailable_during_teardown',
                             owned_actor_count=len(actors))
        if success and cleaned:
            journal.emit('result', 'pass')
    return 0 if success and cleaned else 1


def _json_args(args, output=None):
    values = vars(args).copy()
    for name in ('output', 'managed_server_manifest'):
        if values.get(name) is not None:
            values[name] = str(output if name == 'output' and output is not None
                               else values[name])
    values.pop('manifest_digest', None)
    values.pop('manifest_nonce', None)
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', help='Omit to keep current map; explicit name permits a map change')
    parser.add_argument('--sensors', type=sensor_names, default=['rgb'])
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=5.)
    parser.add_argument('--budget', type=float, default=90., help='Hard worker wall-clock limit, including cleanup')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--fps', type=int, default=40)
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--traffic-manager', action='store_true')
    parser.add_argument('--tm-mode', choices=['sync-only', 'autopilot'], default=None,
                        help='With --traffic-manager: attach/seed only, or also enable ego autopilot (legacy default)')
    parser.add_argument('--ego-motion', choices=['stationary', 'manual-forward'],
                        default='stationary',
                        help='Controlled ego motion for non-autopilot diagnostics')
    parser.add_argument('--manual-throttle', type=float, default=0.2,
                        help='Fixed throttle for --ego-motion manual-forward')
    parser.add_argument('--heavy-vehicles', type=int, default=0)
    parser.add_argument('--two-wheelers', type=int, default=0)
    parser.add_argument('--walkers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--spawn-order', choices=['sensors-first', 'actors-first'], default='sensors-first')
    parser.add_argument('--output', type=Path, help='NEW diagnostic directory, not a dataset directory')
    parser.add_argument('--managed-server-manifest', type=Path,
                        help='Reviewed one-use manifest for launcher-managed evidence')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--manifest-digest', help=argparse.SUPPRESS)
    parser.add_argument('--manifest-nonce', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.tm_mode is not None and not args.traffic_manager:
        parser.error('--tm-mode requires --traffic-manager')
    if args.tm_mode == 'sync-only' and (args.heavy_vehicles or args.two_wheelers or args.walkers):
        parser.error('--tm-mode sync-only requires zero helper actors to isolate TM attachment')
    args.tm_mode = args.tm_mode or 'autopilot'
    if args.tm_mode == 'autopilot' and args.ego_motion != 'stationary':
        parser.error('--ego-motion manual-forward conflicts with --tm-mode autopilot')
    if (min(args.frames, args.width, args.height, args.fps) <= 0
            or not math.isfinite(args.timeout) or not 0 < args.timeout <= 30
            or not math.isfinite(args.budget) or not args.timeout < args.budget <= 600
            or not math.isfinite(args.manual_throttle) or not 0 < args.manual_throttle <= 1.0
            or min(args.heavy_vehicles, args.two_wheelers, args.walkers, args.seed) < 0):
        parser.error('Invalid positive dimensions/frame limits, timeout (0,30], budget (timeout,600], or counts')
    if (args.heavy_vehicles or args.two_wheelers) and not args.traffic_manager:
        parser.error('Vehicle helper isolation requires explicit --traffic-manager')
    if args.managed_server_manifest is not None and not args.town:
        parser.error('--managed-server-manifest requires an explicit --town')
    internal_claim = args.manifest_digest is not None or args.manifest_nonce is not None
    if internal_claim and not args.worker:
        parser.error('Managed claim fields are internal worker arguments')
    if args.worker and args.managed_server_manifest is not None and not (
            args.manifest_digest and args.manifest_nonce):
        parser.error('Managed worker requires its manifest claim')
    if args.worker and args.managed_server_manifest is None and internal_claim:
        parser.error('Managed claim requires --managed-server-manifest')
    if args.dry_run:
        print(json.dumps(_json_args(args), indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required outside dry-run')
    output = args.output.resolve()
    if args.worker:
        with (output / 'events.jsonl').open('x', encoding='utf-8') as stream:
            journal = ProbeJournal(stream)
            if args.managed_server_manifest is not None:
                try:
                    evidence = managed_host_check(
                        args.managed_server_manifest.resolve(), host=args.host,
                        port=args.port, expected_digest=args.manifest_digest,
                        nonce=args.manifest_nonce)
                    journal.emit('host.preworkload', 'verified', **evidence['host'])
                except Exception as exc:
                    journal.emit('host.preworkload', 'error',
                                 error=f'{type(exc).__name__}: {exc}')
                    return 1
            return worker(args, journal)

    managed = None
    claim_nonce = None
    if args.managed_server_manifest is not None:
        from modules.carla_host_guard import claim_manifest, marker_path
        args.managed_server_manifest = args.managed_server_manifest.resolve()
        if marker_path(args.managed_server_manifest).exists():
            print('Managed-server manifest was already consumed', file=sys.stderr)
            return 2
        try:
            managed = managed_host_check(args.managed_server_manifest,
                                         host=args.host, port=args.port)
        except Exception as exc:
            print(f'Managed-server preflight failed: {type(exc).__name__}: {exc}',
                  file=sys.stderr)
            return 2
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(
        json.dumps(_json_args(args, output), indent=2), encoding='utf-8')
    if managed is not None:
        (output / 'host_preflight.json').write_text(
            json.dumps({'manifest_sha256': managed['manifest_sha256'],
                        **managed['host']}, indent=2), encoding='utf-8')
        try:
            _, claim_nonce = claim_manifest(
                args.managed_server_manifest, managed['manifest_sha256'],
                managed['manifest']['run_id'])
        except Exception as exc:
            report = {'status': 'FAIL', 'phase': 'manifest_claim',
                      'error': f'{type(exc).__name__}: {exc}',
                      'carla_shutdown_requested': False}
            (output / 'report.json').write_text(json.dumps(report, indent=2),
                                                encoding='utf-8')
            print(json.dumps(report, indent=2))
            return 1
    before = crash_snapshot()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:]), '--worker']
    if managed is not None:
        command.extend(['--manifest-digest', managed['manifest_sha256'],
                        '--manifest-nonce', claim_nonce])
    supervision = None
    try:
        with (output / 'worker.log').open('x', encoding='utf-8') as log:
            supervision = supervise_worker(command, log, args.budget)
        code, timed_out, interrupted = supervision
    except Exception as exc:
        code, timed_out, interrupted = None, False, False
        launch_error = f'{type(exc).__name__}: {exc}'
    else:
        launch_error = None
    report = summarize_probe(
        read_events(output / 'events.jsonl'), code, timed_out,
        required_sensors=args.sensors, requested_frames=args.frames)
    report['interrupted'] = interrupted
    report['worker_termination'] = getattr(
        supervision, 'termination',
        {'forced': False, 'reason': None, 'kill_attempted': False,
         'kill_succeeded': None, 'exit_verified': False,
         'error': launch_error})
    report['carla_shutdown_requested'] = False
    if interrupted:
        report.update(status='FAIL', requires_world_check=True)
    report['new_crash_reports'] = new_crash_reports(before, crash_snapshot())
    report['scope'] = 'startup/frame transport/cleanup only; NOT label quality or safety acceptance'
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
