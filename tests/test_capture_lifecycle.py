"""Exercise the collector lifecycle with an in-memory world; never connect CARLA."""
from contextlib import ExitStack, redirect_stdout
from copy import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import carla
import config
import numpy as np

import scripts.dataset.collect_carla_dataset as capture
import scripts.dataset.validate_dataset as validate_dataset
from modules import ego_control, simulation_guard, traffic_spawner


class CaptureLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.base = self.root / 'train/Town02/clear'
        self.stack.enter_context(redirect_stdout(StringIO()))
        self.stack.enter_context(patch.object(config, 'CAM_WIDTH', 8))
        self.stack.enter_context(patch.object(config, 'CAM_HEIGHT', 4))
        self.settings = SimpleNamespace(synchronous_mode=False,
                                        fixed_delta_seconds=None)
        self.weather = carla.WeatherParameters()
        self.frame = 0
        self.sensors = []
        self.blueprints = {}
        self.owned = {}
        self.drop_radar = False
        self.listen_error = False
        self.nan_timestamp = False
        self.nan_lidar = False
        self.tick_jump = False
        self.ego = Mock(id=1, type_id='vehicle.test', attributes={})
        self.ego.get_location.return_value = carla.Location()
        self.owned[1] = self.ego
        self.world = Mock(id=10)
        self.world.get_settings.side_effect = lambda: copy(self.settings)
        self.world.apply_settings.side_effect = self.apply_settings
        self.world.get_weather.side_effect = lambda: self.weather
        self.world.set_weather.side_effect = lambda value: setattr(self, 'weather', value)
        self.world.get_map.return_value = SimpleNamespace(
            name='Town02', get_spawn_points=lambda: [carla.Transform()])
        self.world.get_environment_objects.return_value = []
        self.world.get_snapshot.side_effect = self.snapshot
        self.world.wait_for_tick.side_effect = lambda _timeout: self.advance_async()
        self.world.tick.side_effect = self.tick
        self.world.spawn_actor.side_effect = self.spawn_sensor
        self.world.get_actors.side_effect = lambda ids=None: (
            [self.owned[i] for i in ids if i in self.owned] if ids else [])
        self.world.get_blueprint_library.return_value.find.side_effect = self.blueprint
        self.client = Mock()
        self.client.get_world.return_value = self.world
        self.client.apply_batch_sync.side_effect = self.destroy
        self.stack.enter_context(patch.object(carla, 'Client', return_value=self.client))
        self.selector = self.stack.enter_context(patch.object(
            simulation_guard, 'select_probe_world',
            return_value=(self.world, False, {'mode': 'same-map'})))
        self.stack.enter_context(patch.object(ego_control, 'spawn_ego_safe',
                                             return_value=(self.ego, 0)))
        self.controller = Mock()
        self.controller.apply.return_value = {'mode': 'custom'}

        def controller_factory(ego, _tm, cfg, world):
            self.assertEqual(cfg.EGO_CONTROL_MODE, 'custom')
            ego.set_autopilot(False)
            return self.controller

        self.stack.enter_context(patch.object(ego_control, 'EgoController',
                                             side_effect=controller_factory))
        self.spawner = self.stack.enter_context(patch.object(
            traffic_spawner, 'TrafficSpawner'))
        self.spawner.return_value.spawn_traffic.return_value = 0
        for name in ('spawn_target_vehicles', 'spawn_heavy_vehicles',
                     'spawn_two_wheelers'):
            self.stack.enter_context(patch.object(capture, name, return_value=[]))
        self.stack.enter_context(patch.object(capture, 'spawn_walkers',
                                             return_value=([], [])))

    def blueprint(self, name):
        if name not in self.blueprints:
            bp = Mock(id=name)
            bp.has_attribute.return_value = True
            self.blueprints[name] = bp
        return self.blueprints[name]

    def apply_settings(self, value, _timeout):
        self.settings = copy(value)
        return self.frame

    def snapshot(self):
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(
            elapsed_seconds=self.frame / 40.0))

    def advance_async(self):
        self.frame += 1
        return self.snapshot()

    def tick(self, timeout):
        self.assertGreater(timeout, 0)
        self.frame += 2 if self.tick_jump and self.frame >= 2 else 1
        for sensor in self.sensors:
            if not sensor.listening:
                continue
            if self.drop_radar and sensor.kind == 'radar' and self.frame >= 3:
                continue
            timestamp = (float('nan') if self.nan_timestamp and self.frame >= 3
                         else self.frame / 40.0)
            image = np.full((config.CAM_HEIGHT, config.CAM_WIDTH, 4),
                            self.frame, dtype=np.uint8)
            data = SimpleNamespace(
                frame=self.frame, timestamp=timestamp,
                width=config.CAM_WIDTH, height=config.CAM_HEIGHT,
                raw_data=(np.full((1, 4), np.nan if self.nan_lidar and sensor.kind == 'lidar'
                                 and self.frame >= 3 else 0, np.float32).tobytes()
                          if sensor.kind in ('lidar', 'radar') else image.tobytes()))
            sensor.callback(data)
        return self.frame

    def spawn_sensor(self, bp, transform, attach_to):
        sensor = Mock(id=20 + len(self.sensors), is_alive=True)
        sensor.kind = ('semantic' if 'semantic' in bp.id else bp.id.rsplit('.', 1)[-1])
        if sensor.kind == 'ray_cast':
            sensor.kind = 'lidar'
        sensor.listening = False
        sensor.is_listening.side_effect = lambda: sensor.listening
        sensor.get_transform.return_value = transform
        sensor.stop.side_effect = lambda: setattr(sensor, 'listening', False)

        def listen(callback):
            if self.listen_error:
                raise RuntimeError('listen failed')
            sensor.callback = callback
            sensor.listening = True

        sensor.listen.side_effect = listen
        self.sensors.append(sensor)
        self.owned[sensor.id] = sensor
        return sensor

    def destroy(self, commands, do_tick):
        # The fixture supplies responses; lifecycle ownership is checked below.
        self.owned.clear()
        if do_tick:
            self.tick(1.0)
        return [SimpleNamespace(has_error=lambda: False) for _ in commands]

    def run_capture(self, *extra):
        return capture.main([
            '--output', str(self.root), '--frames', '2', '--min-free-gb', '0',
            '--npcs', '0', '--walkers', '0', '--two-wheelers', '0',
            '--heavy-vehicles', '0', *extra])

    def attempt(self):
        files = list((self.base / 'attempts').glob('*.json'))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding='utf-8'))

    def test_complete_capture_metadata_files_and_cleanup(self):
        self.assertEqual(self.run_capture(), 0)
        manifest = json.loads((self.base / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'completed')
        self.assertTrue(manifest['cleanup']['verified'])
        rows = [json.loads(line) for line in
                (self.base / 'annotations.jsonl').read_text().splitlines()]
        self.assertEqual([r['frame_id'] for r in rows], [3, 4])
        for row in rows:
            self.assertEqual(set(row['sensor_frames'].values()), {row['frame_id']})
            self.assertEqual(set(row['sensor_timestamps_s'].values()), {row['timestamp']})
            self.assertEqual(row['episode_id'], 10)
            self.assertEqual(row['ego_control_owner'], 'custom')
        self.ego.set_autopilot.assert_called_once_with(False)
        self.assertTrue(validate_dataset.inspect_dataset(self.root)['valid'])
        self.assertFalse(self.owned)
        self.assertFalse(self.settings.synchronous_mode)
        self.client.get_trafficmanager.return_value.set_synchronous_mode.assert_called_with(False)
        self.assertEqual(self.attempt()['status'], 'completed')

    def test_no_map_selection_when_world_is_owned(self):
        self.settings.synchronous_mode = True
        with self.assertRaisesRegex(RuntimeError, 'synchronous client'):
            self.run_capture()
        self.selector.assert_not_called()
        self.world.apply_settings.assert_not_called()
        self.assertEqual(self.attempt()['status'], 'failed')

    def test_sensor_owned_before_listen_failure(self):
        self.listen_error = True
        with self.assertRaisesRegex(RuntimeError, 'listen failed'):
            self.run_capture()
        attempt = self.attempt()
        self.assertEqual(attempt['cleanup']['actor_destroy']['requested_ids'], [20, 1])
        self.assertTrue(attempt['cleanup']['verified'])
        self.assertFalse((self.base / 'manifest.json').exists())

    def test_actor_shortfall_fails_before_sensors(self):
        with self.assertRaisesRegex(RuntimeError, 'NPC shortfall'):
            self.run_capture('--npcs', '1')
        self.world.spawn_actor.assert_not_called()
        self.assertEqual(self.attempt()['status'], 'failed')

    def test_explicit_ego_modes(self):
        self.run_capture('--ego-control', 'stationary')
        self.ego.set_autopilot.assert_called_once_with(False)
        self.controller.apply.assert_not_called()

    def test_only_experimental_mode_enables_ego_autopilot(self):
        self.run_capture('--ego-control', 'experimental-autopilot')
        self.assertIs(self.ego.set_autopilot.call_args.args[0], True)
        self.controller.apply.assert_not_called()

    def test_custom_fault_fails_without_autopilot(self):
        self.controller.apply.return_value = {'mode': 'custom_fault_safe_stop'}
        with self.assertRaisesRegex(RuntimeError, 'authorized mode'):
            self.run_capture()
        self.ego.set_autopilot.assert_called_once_with(False)
        self.assertEqual(self.attempt()['status'], 'failed')

    def test_optional_radar_absence_is_explicit(self):
        self.drop_radar = True
        self.run_capture('--radar-policy', 'optional')
        rows = [json.loads(line) for line in
                (self.base / 'annotations.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['sensor_frames']['radar'] is None for row in rows))
        self.assertTrue(all(row['radar_available'] is False for row in rows))

    def test_required_radar_failure_retains_counts(self):
        self.drop_radar = True
        with self.assertRaises(TimeoutError):
            self.run_capture()
        attempt = self.attempt()
        self.assertEqual(attempt['status'], 'failed')
        self.assertEqual(attempt['sensor_counts']['radar']['missing'], 1)
        self.assertFalse((self.base / 'manifest.json').exists())

    def test_nonfinite_sensor_timestamp_rejected_before_save(self):
        self.nan_timestamp = True
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.run_capture()
        self.assertFalse(list((self.base / 'rgb').glob('*.jpg')))

    def test_nonfinite_range_data_cannot_complete_capture(self):
        self.nan_lidar = True
        with self.assertRaisesRegex(ValueError, 'lidar'):
            self.run_capture()
        self.assertFalse(list((self.base / 'rgb').glob('*.jpg')))
        self.assertEqual(self.attempt()['sensor_counts']['lidar']['invalid'], 1)

    def test_skipped_world_frame_rejected_before_save(self):
        self.tick_jump = True
        with self.assertRaisesRegex(RuntimeError, 'frame'):
            self.run_capture()
        self.assertFalse(list((self.base / 'rgb').glob('*.jpg')))

    def test_final_attempt_write_failure_cannot_publish_completed_manifest(self):
        real_write = capture.write_json_atomic

        def fail_final(path, value):
            if Path(path).parent.name == 'attempts' and value.get('status') == 'completed':
                raise OSError('final attempt failed')
            return real_write(path, value)

        with patch.object(capture, 'write_json_atomic', side_effect=fail_final):
            with self.assertRaisesRegex(OSError, 'final attempt failed'):
                self.run_capture()
        self.assertFalse((self.base / 'manifest.json').exists())

    def test_cleanup_failure_preserves_original_exception(self):
        self.listen_error = True
        def fail_disable(enabled):
            if not enabled:
                raise RuntimeError('TM disable failed')
        self.client.get_trafficmanager.return_value.set_synchronous_mode.side_effect = fail_disable
        with self.assertRaisesRegex(RuntimeError, 'listen failed') as caught:
            self.run_capture()
        self.assertTrue(any('TM disable failed' in note
                            for note in caught.exception.__notes__))
        self.assertFalse(self.attempt()['cleanup']['verified'])
        self.assertFalse(self.settings.synchronous_mode)

    def test_all_cleanup_verifications_attempted_after_settings_query_failure(self):
        original = copy(self.settings)
        self.world.get_settings.side_effect = RuntimeError('settings read failed')
        result = capture.cleanup_capture(
            client=self.client, world=self.world, session=None,
            traffic_manager=None, traffic_manager_touched=False,
            actors=[self.ego], sensors={}, original_settings=original,
            original_weather=self.weather, world_settings_touched=True,
            weather_touched=True, carla_module=carla)
        self.assertFalse(result['verified'])
        self.world.get_weather.assert_called_once()
        self.world.get_actors.assert_called_once_with([1])

    def test_missing_required_cleanup_handles_fail_closed(self):
        for stage in ('traffic_manager_touched', 'weather_touched', 'world_settings_touched'):
            with self.subTest(stage=stage):
                flags = dict(traffic_manager_touched=False,
                             weather_touched=False, world_settings_touched=False)
                flags[stage] = True
                result = capture.cleanup_capture(
                    client=None, world=None, session=None, traffic_manager=None,
                    actors=[], sensors={}, original_settings=None,
                    original_weather=None, carla_module=carla, **flags)
                self.assertFalse(result['verified'])

    def test_every_cleanup_stage_failure_blocks_success(self):
        for stage in ('stop', 'destroy', 'tm', 'weather', 'settings', 'progress', 'survivor'):
            with self.subTest(stage=stage):
                sensor = Mock(id=20, is_alive=True)
                sensor.is_listening.return_value = False
                world = Mock()
                world.get_settings.return_value = copy(self.settings)
                world.get_weather.return_value = self.weather
                world.get_actors.return_value = []
                world.get_snapshot.return_value = self.snapshot()
                world.wait_for_tick.return_value = SimpleNamespace(
                    frame=1, timestamp=SimpleNamespace(elapsed_seconds=0.025))
                tm, client, session = Mock(), Mock(), Mock(active=True)
                client.apply_batch_sync.return_value = [SimpleNamespace(has_error=lambda: False)]
                target = {'stop': sensor.stop, 'destroy': client.apply_batch_sync,
                          'tm': tm.set_synchronous_mode, 'weather': world.set_weather,
                          'settings': session.close, 'progress': world.wait_for_tick,
                          'survivor': world.get_actors}[stage]
                target.side_effect = RuntimeError(stage + ' failed')
                result = capture.cleanup_capture(
                    client=client, world=world, session=session, traffic_manager=tm,
                    traffic_manager_touched=True, actors=[sensor], sensors={'rgb': sensor},
                    original_settings=self.settings, original_weather=self.weather,
                    world_settings_touched=True, weather_touched=True, carla_module=carla)
                self.assertFalse(result['verified'])
                for call in (sensor.stop, client.apply_batch_sync,
                             tm.set_synchronous_mode, world.set_weather, session.close,
                             world.wait_for_tick, world.get_settings,
                             world.get_weather, world.get_actors):
                    call.assert_called_once()

    def test_config_reaches_blueprints_and_full_pose(self):
        with patch.object(config, 'CAM_ROLL', 1.0), \
                patch.object(config, 'LIDAR_PITCH', 2.0), \
                patch.object(config, 'RADAR_YAW', 3.0):
            self.run_capture()
        for name in ('sensor.camera.rgb', 'sensor.camera.semantic_segmentation',
                     'sensor.camera.depth'):
            calls = dict(call.args for call in self.blueprints[name].set_attribute.call_args_list)
            self.assertEqual(calls['image_size_x'], '8')
            self.assertEqual(calls['image_size_y'], '4')
            self.assertEqual(calls['enable_postprocess_effects'], 'True')
            self.assertEqual(calls['sensor_tick'], '0.0')
        poses = {sensor.kind: sensor.get_transform.return_value for sensor in self.sensors}
        self.assertEqual(poses['rgb'].rotation.roll, 1.0)
        self.assertEqual(poses['lidar'].rotation.pitch, 2.0)
        self.assertEqual(poses['radar'].rotation.yaw, 3.0)

    def test_interrupt_during_image_write_rolls_back_files(self):
        import cv2
        real_write = cv2.imwrite
        calls = []
        def interrupt(path, array, *args):
            if calls:
                raise KeyboardInterrupt('interrupted write')
            calls.append(path)
            return real_write(path, array, *args)
        with patch.object(cv2, 'imwrite', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_capture()
        self.assertEqual(self.attempt()['status'], 'failed')
        self.assertFalse(list((self.base / 'rgb').iterdir()))
        self.assertEqual((self.base / 'annotations.jsonl').read_text(), '')

    def test_corrupted_modality_blocks_completed_resume(self):
        self.run_capture()
        next((self.base / 'semantic').glob('*.png')).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'unreadable'):
            self.run_capture()

    def test_rename_failure_rolls_back_entire_sample(self):
        replace = capture.os.replace
        def fail_depth(source, destination):
            if Path(destination).parent.name == 'depth':
                raise OSError('depth rename failed')
            return replace(source, destination)
        with patch.object(capture.os, 'replace', side_effect=fail_depth):
            with self.assertRaisesRegex(OSError, 'depth rename failed'):
                self.run_capture()
        for name in validate_dataset.MODALITIES:
            self.assertFalse(list((self.base / name).iterdir()))
        self.assertEqual((self.base / 'annotations.jsonl').read_text(), '')
        self.assertEqual(self.attempt()['counters']['write_errors'], 1)

    def test_label_failure_rolls_back_line_and_modalities(self):
        real_append = capture.append_sample_record
        def fail_label(stream, record, paths):
            record['unserializable'] = object()
            return real_append(stream, record, paths)
        with patch.object(capture, 'append_sample_record', side_effect=fail_label):
            with self.assertRaises(TypeError):
                self.run_capture()
        for name in validate_dataset.MODALITIES:
            self.assertFalse(list((self.base / name).iterdir()))
        self.assertEqual((self.base / 'annotations.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
