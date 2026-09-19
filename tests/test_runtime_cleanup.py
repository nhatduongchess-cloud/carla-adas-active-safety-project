"""Offline S1.5 failure-containment checks; no CARLA connection."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from modules.simulation_guard import restore_world_settings


class RestoreFailureTests(unittest.TestCase):
    def test_tm_failure_does_not_skip_world_restore_or_report_success(self):
        world, tm = Mock(), Mock()
        tm.set_synchronous_mode.side_effect = RuntimeError('TM restore failed')
        with self.assertRaisesRegex(RuntimeError, 'TM restore failed'):
            restore_world_settings(world, object(), traffic_manager=tm)
        world.apply_settings.assert_called_once()


class RuntimeCleanupTests(unittest.TestCase):
    def fixture(self):
        actor = Mock(id=1, is_alive=True)
        actor.destroy.return_value = True
        world = Mock()
        world.id = 7
        world.get_actors.return_value = []
        world.get_settings.return_value = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        world.get_snapshot.return_value = SimpleNamespace(frame=10)
        world.wait_for_tick.return_value = SimpleNamespace(frame=11)
        client = Mock()
        client.get_world.return_value = world
        return client, world, actor

    def run_cleanup(self, client, world, actor, **kwargs):
        from modules.runtime_cleanup import cleanup_runtime
        return cleanup_runtime(client=client, world=world, actors=[actor],
                               original_settings=SimpleNamespace(synchronous_mode=False,
                                                                  fixed_delta_seconds=None),
                               **kwargs)

    def test_success_requires_actor_and_async_progress_verification(self):
        client, world, actor = self.fixture()
        result = self.run_cleanup(client, world, actor)
        self.assertTrue(result['verified'])
        actor.destroy.assert_called_once()
        world.tick.assert_not_called()

    def test_sensor_stop_failure_preserves_primary_and_independent_destroy(self):
        client, world, actor = self.fixture()
        actor.stop.side_effect = RuntimeError('stop failed')
        rig = SimpleNamespace(actors=[actor])
        result = self.run_cleanup(client, world, actor, sensor_rig=rig, primary_error='original failure')
        self.assertFalse(result['verified'])
        self.assertEqual(result['primary_error'], 'original failure')
        actor.destroy.assert_called_once()

    def test_dead_server_skips_actor_rpcs_but_closes_local_owners(self):
        client, world, actor = self.fixture()
        client.get_world.side_effect = RuntimeError('server gone')
        telemetry = Mock()
        result = self.run_cleanup(client, world, actor, telemetry=telemetry)
        self.assertFalse(result['verified'])
        actor.destroy.assert_not_called()
        telemetry.close.assert_called_once()

    def test_surviving_actor_or_failed_destroy_is_not_success(self):
        for alive in (False, True):
            client, world, actor = self.fixture()
            actor.destroy.return_value = not alive
            world.get_actors.return_value = [actor]
            self.assertFalse(self.run_cleanup(client, world, actor)['verified'])

    def test_log_error_does_not_skip_cleanup(self):
        client, world, actor = self.fixture()
        journal = Mock()
        journal.close.side_effect = UnicodeError('log failure')
        result = self.run_cleanup(client, world, actor, journal=journal)
        actor.destroy.assert_called_once()
        self.assertFalse(result['verified'])

    def test_unfinished_neural_worker_fails_cleanup(self):
        client, world, actor = self.fixture()
        neural = Mock()
        neural.scheduler_stats.return_value = {'thread_alive': True, 'lane': None}
        self.assertFalse(self.run_cleanup(client, world, actor, perception=neural)['verified'])

    def test_settings_mismatch_and_no_progress_fail(self):
        for defect in ('settings', 'progress'):
            client, world, actor = self.fixture()
            if defect == 'settings':
                world.get_settings.return_value.fixed_delta_seconds = .025
            else:
                world.wait_for_tick.return_value.frame = 10
            self.assertFalse(self.run_cleanup(client, world, actor)['verified'])

    def test_dead_server_abandons_session_without_atexit_retry(self):
        from modules.simulation_guard import SynchronousWorldSession
        client, world, actor = self.fixture()
        session = SynchronousWorldSession(world, 40)
        session.active = True
        client.get_world.side_effect = RuntimeError('server gone')
        result = self.run_cleanup(client, world, actor, session=session)
        self.assertFalse(result['verified'])
        self.assertFalse(session.active)
        world.apply_settings.assert_not_called()

    def test_false_sensor_stop_does_not_cancel_other_owned_steps(self):
        client, world, actor = self.fixture()
        actor.stop.return_value = False
        result = self.run_cleanup(client, world, actor, sensor_rig=SimpleNamespace(actors=[actor]))
        self.assertFalse(result['verified'])
        actor.destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
