"""Pure offline probe tests: no server or model execution."""
import argparse
import copy
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from modules.probe_journal import ProbeJournal, read_events, summarize_probe
from scripts.tools.smoke_carla_stack import (OwnedWorld, ego_state_snapshot, main,
                               managed_host_check, new_crash_reports,
                               sensor_names, supervise_worker, worker)


def valid_capture_events():
    events = [{'stage': 'map', 'event': 'info', 'name': 'Town02',
               'episode_id': 7, 'reloaded': False}]
    for index, frame in enumerate((10, 11)):
        events.append({'stage': 'capture.tick', 'event': 'frame',
                       'index': index, 'world_frame': frame, 'episode_id': 7})
        for sensor in ('rgb', 'lidar'):
            events.append({'stage': f'capture.{sensor}', 'event': 'frame',
                           'index': index,
                           'world_frame': frame, 'sensor_frame': frame,
                           'episode_id': 7})
    events.extend([
        {'stage': 'capture', 'event': 'complete', 'frames': 2,
         'required_sensors': ['rgb', 'lidar'], 'episode_id': 7},
        {'stage': 'cleanup', 'event': 'verified'},
        {'stage': 'result', 'event': 'pass'},
    ])
    return events


class ManagedHostGuardTests(unittest.TestCase):
    def fixture(self):
        from modules.carla_host_guard import build_managed_manifest
        install = Path(__file__).resolve().parents[2]
        launcher = install / 'CarlaUE4.exe'
        shipping = install / 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
        snapshot = {
            'schema_version': 1,
            'powershell_version': '5.1.19041.1',
            'timestamp_format': 'explicit-iso-8601',
            'processes': [
                {'name': 'CarlaUE4.exe', 'pid': 100, 'parent_pid': 50,
                 'exe': str(launcher), 'created_utc': '2026-09-10T10:00:00.0000000Z',
                 'command_line': f'"{launcher}" -quality-level=Low'},
                {'name': 'CarlaUE4-Win64-Shipping.exe', 'pid': 101, 'parent_pid': 100,
                 'exe': str(shipping), 'created_utc': '2026-09-10T10:00:01.0000000Z',
                 'command_line': f'"{shipping}" CarlaUE4 -quality-level=Low'},
            ],
            'listeners': [
                {'local_address': '0.0.0.0', 'local_port': port,
                 'pid': 101, 'state': 'Listen'}
                for port in (2000, 2001, 2002)
            ],
            'connections': [],
        }
        digests = {str(launcher): 'a' * 64, str(shipping): 'b' * 64}
        hash_file = lambda path: digests[str(path)]
        manifest = build_managed_manifest(
            snapshot, install, run_id='S11-fixture', hash_file=hash_file)
        return install, snapshot, manifest, hash_file

    def validate(self, snapshot=None, manifest=None, allowed=()):
        from modules.carla_host_guard import validate_managed_server
        install, base_snapshot, base_manifest, hash_file = self.fixture()
        return validate_managed_server(
            manifest or base_manifest, snapshot or base_snapshot,
            expected_install_root=install,
            project_root=Path(__file__).resolve().parents[1],
            allowed_python_pids=set(allowed), hash_file=hash_file)

    def test_exact_pair_identity_arguments_hash_and_ports_pass(self):
        result = self.validate()
        self.assertEqual((result['launcher_pid'], result['shipping_pid']), (100, 101))
        self.assertEqual(result['listener_ports'], [2000, 2001, 2002])

    def test_process_pair_identity_changes_fail_closed(self):
        _, snapshot, manifest, _ = self.fixture()
        cases = {}
        value = copy.deepcopy(snapshot); value['processes'].append(copy.deepcopy(value['processes'][0])); value['processes'][-1]['pid'] = 102
        cases['extra launcher'] = value
        value = copy.deepcopy(snapshot); value['processes'].append(copy.deepcopy(value['processes'][1])); value['processes'][-1]['pid'] = 102
        cases['extra Shipping'] = value
        value = copy.deepcopy(snapshot); value['processes'][1]['parent_pid'] = 50
        cases['orphan Shipping'] = value
        value = copy.deepcopy(snapshot); value['processes'][1]['exe'] += '.changed'
        cases['wrong path'] = value
        value = copy.deepcopy(snapshot); value['processes'][1]['created_utc'] = '2026-09-10T10:00:02Z'
        cases['PID reuse'] = value
        value = copy.deepcopy(snapshot); value['processes'][1]['command_line'] += ' -d3d11'
        cases['changed arguments'] = value
        for name, changed in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.validate(changed, manifest)

    def test_hash_port_and_peer_python_changes_fail_closed(self):
        install, snapshot, manifest, hash_file = self.fixture()
        from modules.carla_host_guard import validate_managed_server
        changed = copy.deepcopy(manifest)
        changed['shipping']['sha256'] = 'c' * 64
        with self.assertRaisesRegex(RuntimeError, 'hash'):
            validate_managed_server(changed, snapshot, expected_install_root=install,
                                    project_root=Path(__file__).resolve().parents[1],
                                    allowed_python_pids=set(), hash_file=hash_file)
        for name, mutate in (
            ('missing port', lambda value: value['listeners'].pop()),
            ('foreign port', lambda value: value['listeners'][0].update(pid=999)),
        ):
            value = copy.deepcopy(snapshot); mutate(value)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.validate(value, manifest)
        peer = copy.deepcopy(snapshot)
        peer['processes'].append({
            'name': 'python.exe', 'pid': 333, 'parent_pid': 1,
            'exe': str(Path(__file__).resolve().parents[1] / '.venvCarLa/python.exe'),
            'created_utc': '2026-09-10T10:00:03Z',
            'command_line': f'python "{Path(__file__).resolve().parents[1] / "other.py"}"',
        })
        with self.assertRaisesRegex(RuntimeError, 'Python'):
            self.validate(peer, manifest)
        self.assertEqual(self.validate(peer, manifest, allowed={333})['shipping_pid'], 101)

    def test_manifest_load_and_single_use_marker_fail_closed(self):
        from modules.carla_host_guard import (claim_manifest,
                                              load_managed_manifest,
                                              verify_claim)
        _, _, manifest, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            loaded, digest = load_managed_manifest(path)
            self.assertEqual(loaded['run_id'], 'S11-fixture')
            marker, nonce = claim_manifest(path, digest, loaded['run_id'])
            self.assertTrue(marker.is_file())
            with self.assertRaises(FileExistsError):
                claim_manifest(path, digest, loaded['run_id'])
            self.assertEqual(verify_claim(path, digest, loaded['run_id'], nonce)['run_id'],
                             'S11-fixture')
            with self.assertRaises(RuntimeError):
                verify_claim(path, digest, loaded['run_id'], 'wrong-nonce')
            bad = Path(directory) / 'bad.json'
            bad.write_text('[]', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_managed_manifest(bad)

    def test_powershell_snapshot_uses_explicit_iso_timestamp_contract(self):
        from modules.carla_host_guard import query_windows_host
        _, snapshot, _, _ = self.fixture()
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(snapshot), stderr='')
        with patch('modules.carla_host_guard.subprocess.run', return_value=completed) as run:
            self.assertEqual(query_windows_host()['timestamp_format'], 'explicit-iso-8601')
        command = ' '.join(run.call_args.args[0])
        self.assertIn("ToUniversalTime().ToString('o',[Globalization.CultureInfo]::InvariantCulture)",
                      command)
        self.assertNotIn('-DateKind', command)

    def test_missing_pair_and_unrelated_python_contract(self):
        _, snapshot, manifest, _ = self.fixture()
        for process_name in ('CarlaUE4.exe', 'CarlaUE4-Win64-Shipping.exe'):
            value = copy.deepcopy(snapshot)
            value['processes'] = [row for row in value['processes']
                                  if row['name'] != process_name]
            with self.subTest(process=process_name), self.assertRaises(RuntimeError):
                self.validate(value, manifest)
        value = copy.deepcopy(snapshot)
        value['processes'].append({
            'name': 'python.exe', 'pid': 444, 'parent_pid': 1,
            'exe': r'C:\Python312\python.exe',
            'created_utc': '2026-09-10T10:00:03Z',
            'command_line': r'python C:\outside\job.py',
        })
        self.assertEqual(self.validate(value, manifest)['shipping_pid'], 101)

    def test_established_client_connection_fails_regardless_of_executable_path(self):
        _, snapshot, manifest, _ = self.fixture()
        snapshot['processes'].append({
            'name': 'python.exe', 'pid': 555, 'parent_pid': 1,
            'exe': r'C:\Python312\python.exe',
            'created_utc': '2026-09-10T10:00:03Z',
            'command_line': r'python C:\outside\carla_workload.py',
        })
        snapshot['connections'].append({
            'local_address': '127.0.0.1', 'local_port': 51000,
            'remote_address': '127.0.0.1', 'remote_port': 2000,
            'pid': 555, 'state': 'Established',
        })
        with self.assertRaisesRegex(RuntimeError, 'connection'):
            self.validate(snapshot, manifest)


class ProbeWorldSelectionTests(unittest.TestCase):
    def states(self):
        before = {'map': 'Carla/Maps/Town10HD_Opt', 'episode_id': 1,
                  'frame': 10, 'simulation_time_s': 1.0}
        after = {'map': 'Carla/Maps/Town02', 'episode_id': 2,
                 'frames': [20, 21], 'simulation_times': [2.0, 2.025]}
        return before, after

    def test_changed_and_same_map_contracts(self):
        from modules.simulation_guard import validate_probe_world_selection
        before, after = self.states()
        self.assertEqual(validate_probe_world_selection(
            before, after, 'Town02', True)['mode'], 'changed-map')
        before = {'map': 'Town02', 'episode_id': 2, 'frame': 19,
                  'simulation_time_s': 1.975}
        self.assertEqual(validate_probe_world_selection(
            before, after, 'Town02', False)['mode'], 'same-map')

    def test_missing_ack_wrong_map_episode_and_clock_fail_closed(self):
        from modules.simulation_guard import validate_probe_world_selection
        before, after = self.states()
        cases = []
        cases.append(('missing acknowledgement', before, after, None))
        value = copy.deepcopy(after); value['map'] = 'Town05'; cases.append(('wrong map', before, value, True))
        value = copy.deepcopy(after); value['episode_id'] = 1; cases.append(('unchanged episode', before, value, True))
        cases.append(('missing reload', before, after, False))
        same_before = {'map': 'Town02', 'episode_id': 2, 'frame': 19, 'simulation_time_s': 1.975}
        cases.append(('same-map reload', same_before, after, True))
        value = copy.deepcopy(after); value['episode_id'] = 3; cases.append(('same-map episode change', same_before, value, False))
        for frame_pair, times in (([20, 20], [2., 2.025]), ([21, 20], [2., 2.025]),
                                  ([20, 21], [2., 2.]), ([20, 21], [2., float('nan')])):
            value = copy.deepcopy(after); value['frames'] = frame_pair; value['simulation_times'] = times
            cases.append(('invalid progress', before, value, True))
        for name, first, second, reloaded in cases:
            with self.subTest(name=name), self.assertRaises((RuntimeError, ValueError)):
                validate_probe_world_selection(first, second, 'Town02', reloaded)


class CaptureIntegrityTests(unittest.TestCase):
    def test_exact_multisensor_sequence_passes(self):
        from modules.probe_journal import validate_capture_events
        events = valid_capture_events()
        integrity = validate_capture_events(events, ['rgb', 'lidar'], 2)
        self.assertTrue(integrity['valid'])
        self.assertEqual(integrity['frames'], [10, 11])
        self.assertEqual(summarize_probe(
            events, 0, required_sensors=['rgb', 'lidar'], requested_frames=2)['status'], 'PASS')

    def test_missing_mismatched_noncontiguous_or_mixed_capture_fails(self):
        cases = {}
        value = valid_capture_events(); value = [row for row in value if row.get('stage') != 'capture.lidar']; cases['missing sensor'] = value
        value = valid_capture_events(); next(row for row in value if row.get('stage') == 'capture.rgb')['sensor_frame'] = 9; cases['frame mismatch'] = value
        for label, replacement in (('duplicate', 10), ('reversed', 9), ('skipped', 12)):
            value = valid_capture_events()
            for row in value:
                if row.get('world_frame') == 11:
                    row['world_frame'] = replacement
                    if 'sensor_frame' in row:
                        row['sensor_frame'] = replacement
            cases[label] = value
        value = valid_capture_events(); next(row for row in value if row.get('stage') == 'capture.lidar')['episode_id'] = 8; cases['mixed episode'] = value
        value = valid_capture_events(); value[0]['episode_id'] = 8; cases['map episode mismatch'] = value
        value = valid_capture_events(); next(row for row in value if row.get('stage') == 'capture.rgb')['index'] = 1; cases['bad index'] = value
        value = valid_capture_events(); value.insert(-3, {'stage': 'capture.radar', 'event': 'frame', 'index': 1, 'world_frame': 11, 'sensor_frame': 11, 'episode_id': 7}); cases['unknown sensor'] = value
        value = [row for row in valid_capture_events()
                 if not (row.get('stage') == 'capture' and row.get('event') == 'complete')]
        cases['missing completion'] = value
        for name, events in cases.items():
            with self.subTest(name=name):
                report = summarize_probe(
                    events, 0, required_sensors=['rgb', 'lidar'], requested_frames=2)
                self.assertEqual(report['status'], 'FAIL')
                self.assertFalse(report['capture_integrity']['valid'])


class MapTimeoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.old = Mock(id=1)
        self.old.get_map.return_value.name = 'Carla/Maps/Town10HD_Opt'
        self.new = Mock(id=2)
        self.new.get_map.return_value.name = 'Carla/Maps/Town02'
        self.new.get_settings.return_value.synchronous_mode = False
        self.new.get_snapshot.return_value = SimpleNamespace(
            frame=20, timestamp=SimpleNamespace(elapsed_seconds=1.0))
        self.new.wait_for_tick.return_value = SimpleNamespace(
            frame=21, timestamp=SimpleNamespace(elapsed_seconds=1.025))
        self.client = Mock()
        self.client.get_world.side_effect = [self.old, self.new, self.new]
        self.error = RuntimeError('time-out of 15000ms while waiting for the simulator')
        self.client.load_world.side_effect = self.error
        self.diagnostics = {}

    def recover(self, **kwargs):
        from modules.simulation_guard import get_or_load_world
        return get_or_load_world(
            self.client, 'Town02', recovery_timeout_s=2.0,
            client_timeout_s=30.0, diagnostics=self.diagnostics, **kwargs)

    def assert_read_only(self):
        self.client.load_world.assert_called_once_with('Town02')
        for world in (self.old, self.new):
            world.tick.assert_not_called()
            world.apply_settings.assert_not_called()
        self.client.reload_world.assert_not_called()
        self.client.set_timeout.assert_called_with(30.0)

    def test_late_completion_requires_progress_and_retains_timeout(self):
        world, reloaded = self.recover()
        self.assertIs(world, self.new)
        self.assertTrue(reloaded)
        self.assertEqual(self.diagnostics['status'], 'late_completion_verified')
        self.assertEqual(self.diagnostics['load_error'], str(self.error))
        self.assertEqual(self.diagnostics['frames'], [20, 21])
        self.assert_read_only()

    def test_wrong_map_or_sync_owner_fails_closed(self):
        for bad_state in ('wrong_map', 'sync'):
            with self.subTest(bad_state=bad_state):
                self.setUp()
                if bad_state == 'wrong_map':
                    self.new.get_map.return_value.name = 'Town05'
                else:
                    self.new.get_settings.return_value.synchronous_mode = True
                with self.assertRaises(RuntimeError) as caught:
                    self.recover()
                self.assertIs(caught.exception.__cause__, self.error)
                self.assertEqual(self.diagnostics['status'], 'unverified')
                self.assert_read_only()

    def test_stalled_reversed_or_invalid_clock_fails(self):
        for frame, seconds in [(20, 1.025), (19, 1.025), (21, 1.0),
                               (21, float('nan')), (21, float('inf'))]:
            with self.subTest(frame=frame, seconds=seconds):
                self.setUp()
                self.new.wait_for_tick.return_value = SimpleNamespace(
                    frame=frame, timestamp=SimpleNamespace(elapsed_seconds=seconds))
                with self.assertRaises(RuntimeError):
                    self.recover()
                self.assert_read_only()

    def test_rpc_failure_retains_original_error_and_restores_timeout(self):
        self.new.wait_for_tick.side_effect = RuntimeError('stream disconnected')
        with self.assertRaises(RuntimeError) as caught:
            self.recover()
        self.assertIs(caught.exception.__cause__, self.error)
        self.assertIn('stream disconnected', str(caught.exception))
        self.assert_read_only()

    def test_episode_change_during_observation_is_rejected(self):
        self.client.get_world.side_effect = [self.old, self.new, Mock(id=3)]
        with self.assertRaisesRegex(RuntimeError, 'episode'):
            self.recover()
        self.assert_read_only()

    def test_deadline_expires_without_load_retry(self):
        with patch('modules.simulation_guard.time.monotonic', side_effect=[0., 3.]):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                self.recover()
        self.assert_read_only()

    def test_slow_rpc_cannot_pass_after_deadline(self):
        clock = [0.0]
        def slow_snapshot():
            clock[0] = 3.0
            return SimpleNamespace(frame=20)
        self.new.get_snapshot.side_effect = slow_snapshot
        with patch('modules.simulation_guard.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                self.recover()
        self.new.wait_for_tick.assert_not_called()
        self.assert_read_only()

    def test_non_timeout_load_error_is_not_recovered(self):
        self.client.load_world.side_effect = RuntimeError('map asset not found')
        with self.assertRaisesRegex(RuntimeError, 'asset not found'):
            self.recover()
        self.assertEqual(self.client.get_world.call_count, 1)
        self.client.set_timeout.assert_not_called()

    def test_legacy_default_still_propagates_timeout(self):
        from modules.simulation_guard import get_or_load_world
        with self.assertRaises(RuntimeError) as caught:
            get_or_load_world(self.client, 'Town02')
        self.assertIs(caught.exception, self.error)
        self.assertEqual(self.client.get_world.call_count, 1)

    def test_same_map_and_acknowledged_load_do_not_observe(self):
        from modules.simulation_guard import get_or_load_world
        self.client.get_world.side_effect = [self.new]
        self.assertEqual(get_or_load_world(self.client, 'Town02'), (self.new, False))
        self.client.load_world.assert_not_called()
        self.client.get_world.side_effect = [self.old]
        self.client.load_world.side_effect = None
        self.client.load_world.return_value = self.new
        self.assertEqual(self.recover(), (self.new, True))
        self.new.wait_for_tick.assert_not_called()
        self.client.set_timeout.assert_not_called()

    def test_invalid_budgets_rejected_before_world_mutation(self):
        from modules.simulation_guard import get_or_load_world
        for recovery, rpc in [(float('nan'), 30), (float('inf'), 30), (-1, 30),
                              (2, None), (2, 0), (2, float('inf'))]:
            with self.subTest(recovery=recovery, rpc=rpc):
                with self.assertRaises(ValueError):
                    get_or_load_world(self.client, 'Town02', recovery_timeout_s=recovery,
                                      client_timeout_s=rpc)
        self.client.get_world.assert_not_called()
        self.client.load_world.assert_not_called()

    def test_final_state_change_fails_closed(self):
        for change in ('map', 'sync'):
            with self.subTest(change=change):
                self.setUp()
                if change == 'map':
                    self.new.get_map.side_effect = [SimpleNamespace(name='Town02'),
                                                    SimpleNamespace(name='Town05')]
                else:
                    self.new.get_settings.side_effect = [
                        SimpleNamespace(synchronous_mode=False),
                        SimpleNamespace(synchronous_mode=True)]
                with self.assertRaises(RuntimeError):
                    self.recover()
                self.assert_read_only()

    def test_rpc_budgets_shrink_and_restore(self):
        clock = [0.0]
        def clock_step():
            clock[0] += 0.01
            return clock[0]
        with patch('modules.simulation_guard.time.monotonic', side_effect=clock_step):
            self.recover()
        budgets = [call.args[0] for call in self.client.set_timeout.call_args_list[:-1]]
        self.assertTrue(all(0 < b < 2.0 for b in budgets))
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertLess(self.new.wait_for_tick.call_args.args[0], 2.0)
        self.assert_read_only()

    def test_failed_timeout_restore_cannot_report_verified(self):
        def set_timeout(seconds):
            if seconds == 30.0:
                raise RuntimeError('restore client timeout failed')
        self.client.set_timeout.side_effect = set_timeout
        with self.assertRaisesRegex(RuntimeError, 'restore client timeout failed'):
            self.recover()
        self.assertEqual(self.diagnostics['status'], 'unverified')
        self.assert_read_only()

    def test_chinh_opts_in_without_changing_load_or_tick_ownership(self):
        import ast
        tree = ast.parse(Path('chinh.py').read_text(encoding='utf-8-sig'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'get_or_load_world']
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords['recovery_timeout_s']), 10.0)
        self.assertEqual(ast.literal_eval(keywords['client_timeout_s']), 30.0)
        self.assertEqual(keywords['diagnostics'].id, 'map_transition')


class ProbeTests(unittest.TestCase):
    def test_journal_preserves_failure_stage_and_flushes(self):
        stream = Mock()
        journal = ProbeJournal(stream)
        with self.assertRaises(RuntimeError):
            with journal.stage('sensor.rgb.spawn'):
                raise RuntimeError('native API wrapper error')
        self.assertEqual(stream.flush.call_count, 2)
        rows = [json.loads(call.args[0]) for call in stream.write.call_args_list]
        self.assertEqual([r['event'] for r in rows], ['begin', 'error'])
        self.assertIn('native API', rows[-1]['error'])

    def test_pass_requires_capture_cleanup_and_successful_exit(self):
        events = [{'stage': 'cleanup', 'event': 'verified'}, {'stage': 'result', 'event': 'pass'}]
        self.assertEqual(summarize_probe(events, 0)['status'], 'PASS')
        self.assertEqual(summarize_probe(events, 1)['status'], 'FAIL')
        self.assertEqual(summarize_probe(events, 0, True)['status'], 'FAIL')
        self.assertEqual(summarize_probe(events[1:], 0)['status'], 'FAIL')
        self.assertEqual(summarize_probe(events[:1], 0)['status'], 'FAIL')

    def test_cleanup_error_cannot_be_hidden_by_result_pass(self):
        events = [{'stage': 'cleanup', 'event': 'error', 'error': 'stop failed'},
                  {'stage': 'cleanup', 'event': 'verified'}, {'stage': 'result', 'event': 'pass'}]
        self.assertEqual(summarize_probe(events, 0)['status'], 'FAIL')

    def test_native_crash_leaves_unfinished_stage(self):
        events = [{'stage': 'sensor.depth.spawn', 'event': 'begin'}]
        report = summarize_probe(events, -1073741819)
        self.assertEqual(report['unfinished_stages'], ['sensor.depth.spawn'])
        self.assertTrue(report['requires_world_check'])

    def test_truncated_final_journal_line_is_tolerated(self):
        path = Mock()
        path.read_text.return_value = '{"stage":"capture"}\n{"stage":'
        self.assertEqual(read_events(path), [{'stage': 'capture'}])

    def test_actor_registered_before_helper_can_fail(self):
        actor = SimpleNamespace(id=99, type_id='walker.test')
        world = Mock(); world.try_spawn_actor.return_value = actor
        owned = []
        proxy = OwnedWorld(world, owned, ProbeJournal(StringIO()))
        self.assertIs(proxy.try_spawn_actor(SimpleNamespace(id='walker.test'), object()), actor)
        self.assertEqual(owned, [actor])
        world.try_spawn_actor.return_value = None
        proxy.try_spawn_actor(SimpleNamespace(id='walker.test'), object())
        self.assertEqual(owned, [actor])

    def test_sensor_cli_rejects_unknown_duplicate_and_no_rgb(self):
        for value in ('', 'depth', 'rgb,wrong', 'rgb,rgb'):
            with self.assertRaises(argparse.ArgumentTypeError):
                sensor_names(value)
        self.assertEqual(sensor_names('rgb,semantic,depth'), ['rgb', 'semantic', 'depth'])

    def test_dry_run_never_launches_worker_or_creates_output(self):
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen') as popen, \
                patch.object(Path, 'mkdir') as mkdir, \
                patch('scripts.tools.smoke_carla_stack.managed_host_check') as host_check:
            with patch('sys.stdout', new_callable=StringIO):
                self.assertEqual(main(['--dry-run']), 0)
                self.assertEqual(main(['--dry-run', '--town', 'Town02',
                                       '--managed-server-manifest', 'unused.json']), 0)
            popen.assert_not_called(); mkdir.assert_not_called(); host_check.assert_not_called()

    def test_invalid_args_never_launch(self):
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen') as popen, patch('sys.stderr', new_callable=StringIO):
            for args in (['--timeout', 'nan'], ['--budget', 'inf'], ['--frames', '0'],
                         ['--heavy-vehicles', '1'], ['--width', '-1']):
                with self.assertRaises(SystemExit):
                    main([*args, '--dry-run'])
            popen.assert_not_called()

    def test_old_crash_report_not_misattributed(self):
        with patch('scripts.tools.smoke_carla_stack.ET.parse') as parse:
            self.assertEqual(new_crash_reports({'old.xml': 1}, {'old.xml': 1}), [])
            parse.assert_not_called()

    def test_tm_mode_cli_preserves_legacy_and_records_sync_only(self):
        for options, mode in ((['--traffic-manager'], 'autopilot'),
                              (['--traffic-manager', '--tm-mode', 'sync-only'], 'sync-only')):
            with self.subTest(mode=mode), patch('sys.stdout', new_callable=StringIO) as output, \
                    patch('scripts.tools.smoke_carla_stack.subprocess.Popen') as popen:
                self.assertEqual(main([*options, '--dry-run']), 0)
                self.assertEqual(json.loads(output.getvalue())['tm_mode'], mode)
                popen.assert_not_called()

    def test_tm_mode_rejects_ambiguous_or_contaminated_trial(self):
        invalid = [['--tm-mode', 'sync-only'], ['--tm-mode', 'autopilot'],
                   ['--traffic-manager', '--tm-mode', 'unknown']]
        invalid.extend(['--traffic-manager', '--tm-mode', 'sync-only', flag, '1']
                       for flag in ('--heavy-vehicles', '--two-wheelers', '--walkers'))
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen') as popen, \
                patch('sys.stderr', new_callable=StringIO):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    main([*options, '--dry-run'])
            popen.assert_not_called()

    def test_ego_motion_cli_rejects_autopilot_conflict_and_bad_throttle(self):
        invalid = [['--traffic-manager', '--ego-motion', 'manual-forward'],
                   ['--ego-motion', 'manual-forward', '--manual-throttle', '0'],
                   ['--ego-motion', 'manual-forward', '--manual-throttle', '1.1']]
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen') as popen, \
                patch('sys.stderr', new_callable=StringIO):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    main([*options, '--dry-run'])
            popen.assert_not_called()

    def test_ego_state_snapshot_records_pose_speed_and_lane(self):
        location = SimpleNamespace(x=1.5, y=-2.0, z=0.2)
        ego = SimpleNamespace(
            get_location=Mock(return_value=location),
            get_velocity=Mock(return_value=SimpleNamespace(x=3.0, y=4.0, z=0.0)),
            get_transform=Mock(return_value=SimpleNamespace(
                rotation=SimpleNamespace(pitch=1.0, yaw=90.0, roll=0.0))),
        )
        carla_map = SimpleNamespace(get_waypoint=Mock(return_value=SimpleNamespace(
            road_id=10, section_id=2, lane_id=-1, is_junction=False)))
        state = ego_state_snapshot(ego, carla_map)
        self.assertEqual(state['position_m'], {'x': 1.5, 'y': -2.0, 'z': 0.2})
        self.assertEqual(state['speed_mps'], 5.0)
        self.assertEqual((state['road_id'], state['section_id'], state['lane_id']), (10, 2, -1))
        self.assertFalse(state['is_junction'])

    def test_supervisor_deadline_kills_only_owned_worker(self):
        proc = Mock(); proc.poll.return_value = None; proc.returncode = -1
        proc.wait.side_effect = [subprocess.TimeoutExpired('owned', 2), -1]
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen', return_value=proc) as popen:
            self.assertEqual(supervise_worker(['owned'], StringIO(), 2), (-1, True, False))
        proc.kill.assert_called_once()
        self.assertFalse(popen.call_args.kwargs['shell'])
        self.assertEqual(proc.wait.call_args_list[-1].kwargs['timeout'], 10)

    def test_supervisor_ctrl_c_does_not_leave_worker_running(self):
        proc = Mock(); proc.poll.return_value = None; proc.returncode = -1
        proc.wait.side_effect = [KeyboardInterrupt(), -1]
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen', return_value=proc):
            self.assertEqual(supervise_worker(['owned'], StringIO(), 2), (-1, False, True))
        proc.kill.assert_called_once()

    def test_supervisor_reports_unverified_owned_worker_kill_without_tree_kill(self):
        proc = Mock(); proc.poll.return_value = None; proc.returncode = None
        proc.wait.side_effect = subprocess.TimeoutExpired('owned', 2)
        proc.kill.side_effect = OSError('access denied')
        with patch('scripts.tools.smoke_carla_stack.subprocess.Popen', return_value=proc):
            result = supervise_worker(['owned'], StringIO(), 2)
        self.assertEqual(tuple(result), (None, True, False))
        self.assertTrue(result.termination['forced'])
        self.assertFalse(result.termination['exit_verified'])
        self.assertIn('access denied', result.termination['error'])
        source = Path('scripts/tools/smoke_carla_stack.py').read_text(encoding='utf-8')
        for forbidden in ('taskkill', 'Stop-Process', 'TerminateProcess', 'killpg'):
            self.assertNotIn(forbidden, source)

    def test_managed_host_check_verifies_endpoint_claim_and_current_chain(self):
        manifest = {'run_id': 'S11', 'endpoint': {'host': '127.0.0.1',
                                                  'ports': [2000, 2001, 2002]}}
        snapshot = {'processes': []}
        with patch('modules.carla_host_guard.load_managed_manifest',
                   return_value=(manifest, 'a' * 64)), \
                patch('modules.carla_host_guard.verify_claim',
                      return_value={'supervisor_pid': 10}) as verify, \
                patch('modules.carla_host_guard.query_windows_host',
                      return_value=snapshot), \
                patch('modules.carla_host_guard.allowed_python_chain',
                      return_value={10, 11}), \
                patch('modules.carla_host_guard.validate_managed_server',
                      return_value={'shipping_pid': 100}) as validate:
            result = managed_host_check('manifest.json', host='127.0.0.1', port=2000,
                                        expected_digest='a' * 64, nonce='secret')
        verify.assert_called_once()
        self.assertEqual(result['host']['shipping_pid'], 100)
        self.assertEqual(validate.call_args.kwargs['allowed_python_pids'], {10, 11})
        with patch('modules.carla_host_guard.load_managed_manifest',
                   return_value=(manifest, 'b' * 64)):
            with self.assertRaisesRegex(RuntimeError, 'digest'):
                managed_host_check('manifest.json', host='127.0.0.1', port=2000,
                                   expected_digest='a' * 64, nonce='secret')

    def test_managed_preflight_failure_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('scripts.tools.smoke_carla_stack.managed_host_check',
                      side_effect=RuntimeError('identity changed')) as host_check, \
                patch('scripts.tools.smoke_carla_stack.supervise_worker') as supervise, \
                patch('sys.stderr', new_callable=StringIO):
            code = main(['--town', 'Town02', '--managed-server-manifest',
                         str(Path(directory) / 'manifest.json'), '--output',
                         str(Path(directory) / 'run')])
        self.assertEqual(code, 2)
        host_check.assert_called_once()
        supervise.assert_not_called()

    def test_managed_worker_rechecks_host_before_importing_workload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('scripts.tools.smoke_carla_stack.managed_host_check',
                      side_effect=RuntimeError('second check failed')) as host_check, \
                patch('scripts.tools.smoke_carla_stack.worker') as workload:
            code = main(['--worker', '--town', 'Town02',
                         '--managed-server-manifest',
                         str(Path(directory) / 'manifest.json'),
                         '--manifest-digest', 'a' * 64,
                         '--manifest-nonce', 'secret', '--output', directory])
            rows = read_events(Path(directory) / 'events.jsonl')
        self.assertEqual(code, 1)
        host_check.assert_called_once()
        workload.assert_not_called()
        self.assertEqual(rows[-1]['stage'], 'host.preworkload')
        self.assertEqual(rows[-1]['event'], 'error')

    def fake_worker(self, stop_raises=False, server_lost=False,
                    server_lost_after_destroy=False, traffic_manager=False, tm_mode='autopilot',
                    ego_motion='stationary', town='Town02'):
        """Run the complete worker against a fake transport, including cleanup."""
        callbacks, frame = [], [0]
        world, client, session = Mock(), Mock(), MagicMock()
        world.get_settings.return_value = SimpleNamespace(synchronous_mode=False)
        world.id = 7
        world.get_actors.return_value = []
        world.get_map.return_value.name = 'Town02'
        world.get_map.return_value.get_waypoint.return_value = SimpleNamespace(
            road_id=1, section_id=0, lane_id=-1, is_junction=False)
        blueprint = SimpleNamespace(id='vehicle.test')
        world.get_blueprint_library.return_value.find.return_value = blueprint
        ego = Mock(); ego.id = 7; ego.type_id = 'vehicle.test'
        ego.destroy.return_value = True
        ego.get_location.return_value = SimpleNamespace(x=1.0, y=2.0, z=0.0)
        ego.get_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        ego.get_transform.return_value = SimpleNamespace(
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0))
        world.try_spawn_actor.return_value = ego
        sensor = Mock(); sensor.id = 8
        sensor.destroy.return_value = not server_lost_after_destroy
        sensor.listen.side_effect = callbacks.append
        if stop_raises:
            sensor.stop.side_effect = RuntimeError('injected stop failure')
        world.spawn_actor.return_value = sensor
        def tick(*args):
            frame[0] += 1
            for callback in callbacks:
                callback(SimpleNamespace(frame=frame[0], raw_data=b'1234'))
            return frame[0]
        world.tick.side_effect = tick
        world.wait_for_tick.return_value = SimpleNamespace(frame=100)
        if server_lost or server_lost_after_destroy:
            calls = [0]
            def get_world():
                calls[0] += 1
                unavailable_after = 2 if server_lost else 3
                if calls[0] >= unavailable_after:
                    raise RuntimeError('connection refused')
                return world
            client.get_world.side_effect = get_world
        else:
            client.get_world.return_value = world
        client.get_server_version.return_value = client.get_client_version.return_value = 'test-version'
        carla = SimpleNamespace(Client=Mock(return_value=client), Transform=lambda *a: a,
                                Location=lambda *a: a, VehicleControl=Mock(side_effect=lambda **kwargs: kwargs))
        smoke = SimpleNamespace(spawn_test_vehicle=lambda w: w.try_spawn_actor(blueprint, None))
        collector = SimpleNamespace(spawn_heavy_vehicles=Mock(), spawn_two_wheelers=Mock(), spawn_walkers=Mock())
        args = SimpleNamespace(host='127.0.0.1', port=2000, town=town, fps=40, timeout=1.,
                               width=64, height=32, postprocess=False, traffic_manager=traffic_manager,
                               tm_mode=tm_mode,
                               ego_motion=ego_motion, manual_throttle=0.2,
                               heavy_vehicles=0, two_wheelers=0, walkers=0, seed=42,
                               sensors=['rgb'], spawn_order='sensors-first', frames=2)
        stream = StringIO()
        with patch.dict('sys.modules', {'carla': carla, 'scripts.tools.smoke_carla_camera': smoke,
                                      'scripts.dataset.collect_carla_dataset': collector}), \
                patch('modules.simulation_guard.get_or_load_world', return_value=(world, False)), \
                patch('modules.simulation_guard.select_probe_world', return_value=(
                    world, False, {'before': {'map': 'Town02', 'episode_id': 7,
                                              'frame': 0, 'simulation_time_s': 0.0},
                                   'after': {'map': 'Town02', 'episode_id': 7,
                                             'frames': [1, 2],
                                             'simulation_times': [0.025, 0.05]},
                                   'mode': 'same-map'})) as selector, \
                patch('modules.simulation_guard.SynchronousWorldSession', return_value=session), \
                patch('modules.sensor_setup.configure_camera_blueprint', side_effect=lambda bp, cfg: bp):
            code = worker(args, ProbeJournal(stream))
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.fake_events = events
        self.fake_sensor, self.fake_ego, self.fake_session = sensor, ego, session
        self.fake_client = client
        self.fake_selector = selector
        if server_lost:
            self.assertEqual(sensor.destroy.call_count, 0, events)
            ego.destroy.assert_not_called()
            session.close.assert_not_called()
        elif server_lost_after_destroy or stop_raises:
            sensor.stop.assert_called_once()
            if server_lost_after_destroy:
                sensor.destroy.assert_called_once()
            else:
                sensor.destroy.assert_not_called()
            ego.destroy.assert_not_called()
            session.close.assert_not_called()
        else:
            self.assertEqual(sensor.destroy.call_count, 1, events)
            ego.destroy.assert_called_once()
            session.close.assert_called_once()
        return code, summarize_probe(events, code,
                                     required_sensors=args.sensors,
                                     requested_frames=args.frames)

    def test_worker_success_requires_real_aligned_reads_and_cleanup(self):
        code, result = self.fake_worker()
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'PASS')
        self.fake_client.get_trafficmanager.assert_not_called()
        self.fake_ego.set_autopilot.assert_not_called()
        self.fake_selector.assert_called_once()
        stages = [(event.get('stage'), event.get('event')) for event in self.fake_events]
        self.assertLess(stages.index(('map', 'info')), stages.index(('sync.enter', 'begin')))

    def test_sync_only_attaches_tm_without_registering_ego(self):
        code, result = self.fake_worker(traffic_manager=True, tm_mode='sync-only')
        self.assertEqual(code, 0)
        tm = self.fake_client.get_trafficmanager.return_value
        self.fake_session.attach_traffic_manager.assert_called_once_with(tm)
        tm.set_random_device_seed.assert_called_once_with(42)
        self.fake_ego.set_autopilot.assert_not_called()
        self.assertTrue(result['cleanup_verified'])
        self.assertTrue(any(e.get('stage') == 'tm.mode' and e.get('mode') == 'sync-only'
                            and e.get('ego_autopilot') is False for e in self.fake_events))

    def test_legacy_tm_still_registers_ego_autopilot(self):
        code, _ = self.fake_worker(traffic_manager=True)
        self.assertEqual(code, 0)
        tm = self.fake_client.get_trafficmanager.return_value
        self.fake_ego.set_autopilot.assert_called_once_with(True, tm.get_port())

    def test_manual_forward_applies_control_and_logs_pre_tick_state(self):
        code, result = self.fake_worker(traffic_manager=True, tm_mode='sync-only',
                                        ego_motion='manual-forward')
        self.assertEqual(code, 0)
        self.assertEqual(self.fake_ego.apply_control.call_count, 2)
        samples = [event for event in self.fake_events
                   if event.get('stage') == 'ego.state' and event.get('phase') == 'capture']
        self.assertEqual([sample['index'] for sample in samples], [0, 1])
        self.assertEqual([sample['speed_mps'] for sample in samples], [0.0, 0.0])
        self.assertTrue(result['cleanup_verified'])

    def test_worker_cleanup_aborts_after_sensor_stop_failure(self):
        code, result = self.fake_worker(stop_raises=True)
        self.assertEqual(code, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['cleanup_verified'])
        self.fake_sensor.destroy.assert_not_called()
        self.fake_ego.destroy.assert_not_called()
        self.fake_session.close.assert_not_called()

    def test_worker_server_loss_skips_blocking_teardown(self):
        code, result = self.fake_worker(server_lost=True)
        self.assertEqual(code, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['cleanup_verified'])
        self.fake_sensor.stop.assert_not_called()
        self.fake_sensor.destroy.assert_not_called()
        self.fake_ego.destroy.assert_not_called()
        self.fake_session.close.assert_not_called()
        self.assertTrue(any(e.get('stage') == 'cleanup.server_probe'
                            and e.get('event') == 'unavailable'
                            for e in self.fake_events))

    def test_worker_server_loss_during_teardown_stops_retry_loop(self):
        code, result = self.fake_worker(server_lost_after_destroy=True)
        self.assertEqual(code, 1)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['cleanup_verified'])
        self.assertTrue(any(e.get('stage') == 'cleanup'
                            and e.get('event') == 'skipped'
                            and e.get('reason') == 'server_unavailable_during_teardown'
                            for e in self.fake_events))


class RuntimeCrashJournalTests(unittest.TestCase):
    def make_journal(self, stream=None, **kwargs):
        from modules.probe_journal import RuntimeCrashJournal
        stream = stream if stream is not None else StringIO()
        journal = RuntimeCrashJournal(stream=stream, **kwargs)
        self.addCleanup(journal.close)
        return journal, stream

    def test_disabled_journal_does_not_sample_or_open(self):
        from modules.probe_journal import RuntimeCrashJournal
        with patch('builtins.open') as opened:
            journal = RuntimeCrashJournal()
            world = Mock()
            journal.record_runtime('sensor.before', world=world, ego=Mock())
            journal.close()
        opened.assert_not_called()
        world.get_snapshot.assert_not_called()

    def test_records_preserve_order_and_nonfinite_values_are_null(self):
        journal, stream = self.make_journal()
        journal.emit('sensor', 'before', value=float('nan'))
        journal.emit('sensor', 'after', value=float('inf'))
        journal.emit('control', 'before', value=1)
        journal.emit('control', 'after', value=2)
        journal.close()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([(r['stage'], r['event']) for r in rows],
                         [('sensor', 'before'), ('sensor', 'after'), ('control', 'before'), ('control', 'after')])
        self.assertIsNone(rows[0]['value'])
        self.assertIsNone(rows[1]['value'])
        self.assertIn('sample_monotonic_s', rows[0])

    def test_record_and_byte_caps_preserve_prefix_without_overwrite(self):
        journal, stream = self.make_journal(max_records=2, max_bytes=2048)
        for i in range(8):
            journal.emit('sample', 'info', index=i)
        journal.close()
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([r['index'] for r in rows], [0, 1])
        self.assertEqual(journal.summary()['dropped'], 6)
        self.assertLessEqual(len(stream.getvalue().encode('utf-8')), 2048)
        tiny, target = self.make_journal(max_bytes=64)
        tiny.emit('large', 'info', message='x' * 2000)
        tiny.close()
        self.assertEqual(target.getvalue(), '')
        self.assertTrue(tiny.summary()['capacity_reached'])

    def test_writer_failure_does_not_escape_into_control(self):
        stream = Mock()
        stream.write.side_effect = OSError('disk unavailable')
        journal, _ = self.make_journal(stream)
        journal.emit('control', 'before')
        control = Mock()
        control()
        journal.close()
        control.assert_called_once()
        self.assertIn('OSError', journal.summary()['writer_error'])

    def test_slow_writer_has_bounded_queue_and_nonblocking_producer(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        stream = Mock()
        def block_write(value):
            entered.set()
            release.wait(2)
        stream.write.side_effect = block_write
        journal, _ = self.make_journal(stream, queue_capacity=1)
        try:
            journal.emit('first', 'info')
            self.assertTrue(entered.wait(1))
            for _ in range(10):
                journal.emit('later', 'info')
            self.assertLessEqual(journal.summary()['pending'], 1)
            self.assertGreater(journal.summary()['dropped'], 0)
            journal.close(timeout=0.01)
            self.assertTrue(journal.summary()['writer_alive'])
        finally:
            release.set()
            journal.close(timeout=1)

    def test_sampling_failure_is_recorded_without_hiding_next_command(self):
        journal, stream = self.make_journal()
        world = Mock()
        world.get_snapshot.side_effect = RuntimeError('snapshot unavailable')
        journal.record_runtime('sensor.before', world=world, ego=Mock())
        control = Mock()
        control()
        journal.close()
        control.assert_called_once()
        row = json.loads(stream.getvalue())
        self.assertIn('snapshot', row['unavailable'])
        self.assertIsNone(row['snapshot_frame'])

    def test_snapshot_pose_clock_and_control_readback_are_explicit(self):
        journal, stream = self.make_journal()
        xyz = SimpleNamespace(x=1., y=2., z=3.)
        actor = Mock()
        actor.get_transform.return_value = SimpleNamespace(location=xyz, rotation=SimpleNamespace(pitch=0.,yaw=90.,roll=0.))
        actor.get_velocity.return_value = xyz
        snapshot = SimpleNamespace(frame=17, timestamp=SimpleNamespace(elapsed_seconds=0.5), find=lambda _: actor)
        world = Mock(); world.get_snapshot.return_value = snapshot
        ego = Mock(id=8)
        ego.get_control.return_value = SimpleNamespace(throttle=0., brake=1., steer=.2, hand_brake=False)
        carla_map = Mock()
        carla_map.get_waypoint.return_value = SimpleNamespace(is_junction=True, road_id=2, lane_id=-1)
        journal.record_runtime('control.after', world=world, ego=ego, carla_map=carla_map,
                               read_control=True, frame_id=16, controller_mode='custom_fault_safe_stop')
        journal.close()
        row = json.loads(stream.getvalue())
        self.assertEqual(row['snapshot_frame'],17)
        self.assertEqual(row['simulation_time_s'],.5)
        self.assertEqual(row['frame_id'],16)
        self.assertEqual(row['pose']['location']['y'],2.)
        self.assertTrue(row['road']['is_junction'])
        self.assertEqual(row['control_readback']['brake'],1.)
        self.assertIn('cached',row['control_readback_semantics'])

    def test_before_control_does_not_request_control_readback(self):
        journal, _ = self.make_journal()
        world = Mock(); world.get_snapshot.side_effect = RuntimeError('unavailable')
        ego = Mock()
        journal.record_runtime('control.before', world=world, ego=ego)
        ego.get_control.assert_not_called()

    def test_path_open_is_exclusive_and_existing_data_is_preserved(self):
        from modules.probe_journal import RuntimeCrashJournal
        with patch('builtins.open', side_effect=FileExistsError('existing')) as opened:
            with self.assertRaises(FileExistsError):
                RuntimeCrashJournal('existing.jsonl')
        self.assertEqual(opened.call_args.args[1], 'x')

    def test_cli_is_opt_in(self):
        from modules.runtime_config import build_argument_parser
        parser = build_argument_parser(SimpleNamespace(NUM_NPC_VEHICLES=0))
        self.assertIsNone(parser.parse_args([]).crash_journal)
        self.assertEqual(parser.parse_args(['--crash-journal','new.jsonl']).crash_journal,'new.jsonl')

    def test_bad_limits_rejected_before_file_open(self):
        from modules.probe_journal import RuntimeCrashJournal
        with patch('builtins.open') as opened:
            for value in (0, -1, True, float('nan'), 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    RuntimeCrashJournal('unused.jsonl', max_records=value)
        opened.assert_not_called()

    def test_flush_failure_is_visible_not_retried(self):
        stream = Mock()
        stream.flush.side_effect = OSError('flush failed')
        journal, _ = self.make_journal(stream)
        journal.emit('test', 'info')
        journal.close()
        self.assertEqual(journal.summary()['writer_error'], 'OSError')
        self.assertFalse(journal.summary()['writer_alive'])
        stream.flush.assert_called_once()

    def test_orchestrator_sensor_and_control_hook_order_without_importing_models(self):
        # Execute the actual three-statement hook blocks from chinh.py with
        # fixtures: avoid importing its models or contacting a simulator.
        import ast
        source = (Path(__file__).resolve().parents[1] / 'chinh.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in ast.walk(main) if isinstance(n, ast.While))
        def is_hook(node, phase):
            return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'record_runtime'
                    and node.value.args[0].value == phase)
        def block(phase):
            i = next(i for i, n in enumerate(loop.body) if is_hook(n, phase))
            return compile(ast.fix_missing_locations(ast.Module(body=loop.body[i:i+3], type_ignores=[])),
                           'chinh.py hook block', 'exec')
        events = []
        log = Mock()
        log.record_runtime.side_effect = lambda phase, **kwargs: events.append(phase)
        sensor = Mock()
        sensor.read.side_effect = lambda timestamp: (events.append('sensor.read') or
            SimpleNamespace(frame_id=12, lidar_timestamp=.3, camera_frame_id=10))
        controller = Mock(mode='custom')
        controller.apply.side_effect = lambda *a, **kw: (events.append('controller.apply') or {'mode':'custom'})
        scope = dict(crash_log=log, world=Mock(), ego_vehicle=Mock(), carla_map=Mock(),
                     frame_count=1, ego_controller=controller, sensor_rig=sensor,
                     capture_timestamp=1., w_frame=12,
                     decision=SimpleNamespace(action='BRAKE',state='AEB',brake=1.),
                     l3={'override':False}, odd={'state':'NORMAL'}, target_kmh=20.,
                     traffic_control={}, turn=None)
        exec(block('sensor.before'), scope)
        exec(block('control.before'), scope)
        self.assertEqual(events, ['sensor.before','sensor.read','sensor.after',
                                  'control.before','controller.apply','control.after'])
        self.assertTrue(log.record_runtime.call_args.kwargs['read_control'])
        # An exception in actual control must not produce a false after-record.
        events.clear()
        controller.apply.side_effect = RuntimeError('control failed')
        with self.assertRaisesRegex(RuntimeError, 'control failed'):
            exec(block('control.before'), scope)
        self.assertEqual(events, ['control.before'])


if __name__ == '__main__':
    unittest.main()
