from contextlib import redirect_stdout
from io import StringIO
import json
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import collect_carla_dataset as collector
import collect_week2_dataset as week2
import validate_dataset
from modules.sensor_sync import warmup_sensor_streams
from modules.traffic_spawner import TrafficSpawner


class _NumberAttribute:
    def __int__(self):
        return 4


class _Blueprint:
    id = "vehicle.test"

    def has_attribute(self, name):
        return name in {"number_of_wheels", "base_type", "role_name"}

    def get_attribute(self, name):
        if name == "base_type":
            return SimpleNamespace(as_str=lambda: "car")
        return _NumberAttribute()

    def set_attribute(self, *_args):
        pass


class CollectorArgumentsTests(unittest.TestCase):
    def test_default_control_and_radar_policy_are_safe_and_explicit(self):
        args = collector.parse_args(["--output", "pilot", "--dry-run"])
        self.assertEqual(args.ego_control, "custom")
        self.assertEqual(args.radar_policy, "required")
        self.assertEqual(args.target_speed_kmh, 20.0)

    def test_invalid_dry_run_is_rejected(self):
        with self.assertRaises(SystemExit):
            collector.parse_args([
                "--output", "pilot", "--frames", "0", "--dry-run"])

    def test_unknown_weather_is_rejected_before_dry_run(self):
        with self.assertRaises(SystemExit):
            collector.parse_args(['--output', 'pilot', '--weather', '../outside', '--dry-run'])


class ImmediateOwnershipTests(unittest.TestCase):
    def test_zero_helpers_do_not_query_world_or_navigation(self):
        world = Mock()
        self.assertEqual(collector._spawn_balanced_vehicles(
            world, Mock(), [], Mock(), 'test', reference_vehicle=Mock()), ([], []))
        self.assertEqual(collector.spawn_walkers(
            world, 0, Mock(), reference_vehicle=Mock()), ([], []))
        self.assertEqual(world.mock_calls, [])

    def test_walker_is_owned_before_control_failure(self):
        import carla
        events = []
        walker = SimpleNamespace(id=9)
        def fail_control(_control):
            events.append('control')
            raise RuntimeError('walker control failed')
        walker.apply_control = fail_control
        world = Mock()
        world.get_blueprint_library.return_value.filter.return_value = [_Blueprint()]
        world.get_random_location_from_navigation.return_value = carla.Location()
        world.try_spawn_actor.return_value = walker
        import random
        with self.assertRaisesRegex(RuntimeError, 'walker control failed'):
            collector.spawn_walkers(
                world, 1, random.Random(42), on_spawn=lambda a: events.append(a.id))
        self.assertEqual(events, [9, 'control'])

    def test_balanced_helper_is_owned_before_autopilot_configuration(self):
        events = []
        actor = SimpleNamespace(id=1, type_id="vehicle.test")
        actor.set_autopilot = lambda *_args: (
            events.append("autopilot"),
            (_ for _ in ()).throw(RuntimeError("configuration failed")),
        )[-1]
        world = SimpleNamespace(
            get_blueprint_library=lambda: SimpleNamespace(
                filter=lambda _pattern: [_Blueprint()]),
            get_map=lambda: SimpleNamespace(get_spawn_points=lambda: [object()]),
            try_spawn_actor=lambda *_args: actor,
        )
        traffic_manager = SimpleNamespace(get_port=lambda: 8000)
        rng = SimpleNamespace(shuffle=lambda _items: None,
                              choice=lambda items: items[0])

        with self.assertRaisesRegex(RuntimeError, "configuration failed"):
            collector._spawn_balanced_vehicles(
                world, traffic_manager, ["car"], rng, "test",
                on_spawn=lambda value: events.append(("owned", value.id)))
        self.assertEqual(events, [("owned", 1), "autopilot"])

    def test_traffic_spawner_is_owned_before_autopilot_configuration(self):
        events = []
        actor = SimpleNamespace(id=7, type_id="vehicle.test")
        actor.set_autopilot = lambda *_args: (
            events.append("autopilot"),
            (_ for _ in ()).throw(RuntimeError("configuration failed")),
        )[-1]
        world = SimpleNamespace(
            get_blueprint_library=lambda: SimpleNamespace(
                filter=lambda _pattern: [_Blueprint()]),
            get_map=lambda: SimpleNamespace(get_spawn_points=lambda: [object()]),
            try_spawn_actor=lambda *_args: actor,
        )
        traffic_manager = SimpleNamespace(
            get_port=lambda: 8000, set_random_device_seed=lambda _seed: None)
        spawner = TrafficSpawner(
            world, traffic_manager, seed=42, strict=True,
            on_spawn=lambda value: events.append(("owned", value.id)))

        with self.assertRaisesRegex(RuntimeError, "configuration failed"):
            spawner.spawn_traffic(1)
        self.assertEqual(events, [("owned", 7), "autopilot"])
        self.assertEqual([value.id for value in spawner.actors], [7])


class BoundedSensorTests(unittest.TestCase):
    def test_queue_overflow_replaces_oldest_and_is_counted(self):
        sensor_queue = queue.Queue(maxsize=1)
        drops = {"rgb": 0}
        collector.enqueue_sensor(sensor_queue, "old", drops, "rgb")
        collector.enqueue_sensor(sensor_queue, "new", drops, "rgb")
        self.assertEqual(sensor_queue.get_nowait(), "new")
        self.assertEqual(drops, {"rgb": 1})

    def test_warmup_passes_a_bounded_tick_timeout(self):
        frames = queue.Queue()
        frames.put(SimpleNamespace(frame=1))
        world = SimpleNamespace(tick=Mock(return_value=1))
        self.assertEqual(
            warmup_sensor_streams(
                world, {"rgb": frames}, attempts=1, timeout=0.1,
                tick_timeout=3.0),
            1)
        world.tick.assert_called_once_with(3.0)


class PersistenceAndValidationTests(unittest.TestCase):
    def _arrays(self):
        return {
            "rgb": np.zeros((4, 6, 3), np.uint8),
            "semantic": np.zeros((4, 6), np.uint8),
            "road_line": np.zeros((4, 6), np.uint8),
            "depth": np.zeros((4, 6), np.uint16),
            "lidar": np.zeros((2, 4), np.float32),
            "radar": np.zeros((1, 4), np.float32),
        }

    def test_partial_file_failure_leaves_no_valid_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self._arrays():
                (root / name).mkdir()

            class FailingCv2:
                IMWRITE_JPEG_QUALITY = 1
                calls = 0

                @classmethod
                def imwrite(cls, path, _array, _options=None):
                    cls.calls += 1
                    Path(path).write_bytes(b"partial")
                    return cls.calls < 2

            with self.assertRaises(IOError):
                collector.persist_capture_files(
                    root, "sample", self._arrays(), 90,
                    cv2_module=FailingCv2)
            self.assertFalse(any(path.is_file() for path in root.rglob("*")))

    def test_validator_rejects_missing_semantic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "train" / "Town02" / "clear"
            stem = "seed0000000042_000000"
            for folder, extension in (
                    ("rgb", "jpg"), ("road_line", "png"),
                    ("depth", "png"), ("lidar", "npz"), ("radar", "npz")):
                (base / folder).mkdir(parents=True, exist_ok=True)
                (base / folder / f"{stem}.{extension}").write_bytes(b"x")
            (base / "annotations.jsonl").write_text(json.dumps({
                "sample_id": stem, "frame_id": 1, "town": "Town02",
                "weather": "clear", "split": "train", "objects": [],
            }) + "\n", encoding="utf-8")

            report = validate_dataset.inspect_dataset(root)
            self.assertFalse(report["valid"])
            self.assertTrue(any(issue["type"] == "missing_files"
                                for issue in report["issues"]))


class Week2SupervisionTests(unittest.TestCase):
    def job(self, **changes):
        job = {
            "job_index": 0,
            "split": "train",
            "town": "Town02",
            "weather": "clear",
            "frames": 2,
            "npcs": 15,
            "walkers": 8,
            "two_wheelers": 4,
            "heavy_vehicles": 2,
            "seed": 42,
        }
        job.update(changes)
        return job

    def completed_manifest(self, job):
        return {
            **{key: value for key, value in job.items() if key != "job_index"},
            **week2.COLLECTOR_PROFILE,
            "job_hash": week2.job_hash(job),
            "status": "completed",
            "total_job_samples": job["frames"],
            "cleanup": {"verified": True},
        }

    def write_samples(self, root, job):
        import cv2
        base = week2.manifest_path(root, job).parent
        rows = []
        arrays = PersistenceAndValidationTests()._arrays()
        for name in arrays:
            (base / name).mkdir(parents=True, exist_ok=True)
        for index in range(job['frames']):
            stem = collector.sample_stem(job['seed'], index)
            collector.persist_capture_files(base, stem, arrays, 90, cv2)
            rows.append({'sample_id': stem, 'job_hash': week2.job_hash(job)})
        (base / 'annotations.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def test_manifest_without_sample_files_cannot_complete_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root, job = Path(directory), self.job()
            path = week2.manifest_path(root, job)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self.completed_manifest(job)), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'sample'):
                week2.verify_completed_job(root, job)

    def test_child_process_has_timeout(self):
        runner = Mock(return_value=SimpleNamespace(returncode=0))
        week2.run_checked(["collector"], timeout_s=123.0, runner=runner)
        runner.assert_called_once_with(
            ["collector"], check=True, timeout=123.0)

    def test_completed_manifest_must_match_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = self.job()
            manifest = root / "train" / "Town02" / "clear" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(self.completed_manifest(job)), encoding="utf-8")
            self.write_samples(root, job)
            self.assertEqual(
                week2.verify_completed_job(root, job)["status"], "completed")
            manifest.write_text(json.dumps({
                **self.completed_manifest(job), "frames": 1,
                "total_job_samples": 1,
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                week2.verify_completed_job(root, job)

    def test_plan_hash_is_canonical_and_changes_with_the_contract(self):
        first = {"jobs": [self.job()], "profile": week2.COLLECTOR_PROFILE}
        reordered = {"profile": week2.COLLECTOR_PROFILE,
                     "jobs": [dict(reversed(list(self.job().items())))]}
        self.assertEqual(week2.plan_hash(first), week2.plan_hash(reordered))
        changed = {"jobs": [self.job(seed=1337)],
                   "profile": week2.COLLECTOR_PROFILE}
        self.assertNotEqual(week2.plan_hash(first), week2.plan_hash(changed))

    def test_resume_state_is_bound_to_the_plan_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection_state.json"
            path.write_text(json.dumps({
                "schema_version": week2.STATE_SCHEMA_VERSION,
                "plan_hash": "stale",
                "completed_jobs": [0],
                "failed_job": None,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different plan"):
                week2.load_state(path, "expected", [0])

    def test_existing_different_plan_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection_plan.json"
            old_plan = {"schema_version": 2, "jobs": [self.job(seed=1337)]}
            old_record = {**old_plan, "plan_hash": week2.plan_hash(old_plan)}
            path.write_text(json.dumps(old_record), encoding="utf-8")
            before = path.read_bytes()
            new_plan = {"schema_version": 2, "jobs": [self.job()]}
            with self.assertRaisesRegex(RuntimeError, "refusing overwrite"):
                week2.ensure_plan(path, new_plan, week2.plan_hash(new_plan))
            self.assertEqual(path.read_bytes(), before)

    def test_exact_completed_manifest_is_reconciled_without_carla(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = week2.build_collection_plan(30)[0]
            path = week2.manifest_path(root, job)
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(self.completed_manifest(job)), encoding="utf-8")
            preflight = Mock()
            self.write_samples(root, job)
            runner = Mock(return_value=None)
            with patch.object(sys, "argv", [
                    "collect_week2_dataset.py", "--output", directory,
                    "--frames", "30", "--max-jobs", "1",
                    "--min-free-gb", "0"]), \
                    patch.object(week2, "carla_preflight", preflight), \
                    patch.object(week2, "run_checked", runner), \
                    redirect_stdout(StringIO()):
                week2.main()
            preflight.assert_not_called()
            state = json.loads(
                (root / "collection_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed_jobs"], [0])
            self.assertEqual(state["validation"]["status"], "completed")

    def test_child_exit_zero_without_manifest_is_not_completed(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "argv", [
                    "collect_week2_dataset.py", "--output", directory,
                    "--frames", "30", "--max-jobs", "1",
                    "--min-free-gb", "1"]), \
                patch.object(week2, "carla_preflight", return_value={}), \
                patch.object(week2, "run_checked", return_value=None):
            with redirect_stdout(StringIO()):
                with self.assertRaisesRegex(RuntimeError, "manifest is missing"):
                    week2.main()
            state = json.loads(
                (Path(directory) / "collection_state.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(state["completed_jobs"], [])
            self.assertEqual(state["failed_job"]["error_type"], "RuntimeError")

    def test_timeout_is_recorded_without_completing_the_job(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sys, "argv", [
                    "collect_week2_dataset.py", "--output", directory,
                    "--frames", "30", "--max-jobs", "1",
                    "--min-free-gb", "1", "--job-timeout-s", "7"]), \
                patch.object(week2, "carla_preflight", return_value={}), \
                patch.object(
                    week2, "run_checked",
                    side_effect=subprocess.TimeoutExpired("collector", 7)):
            with redirect_stdout(StringIO()):
                with self.assertRaises(subprocess.TimeoutExpired):
                    week2.main()
            state = json.loads(
                (Path(directory) / "collection_state.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(state["completed_jobs"], [])
            self.assertEqual(state["failed_job"]["timeout_s"], 7.0)

    def test_validator_timeout_is_bounded_and_keeps_job_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = week2.build_collection_plan(30)[0]
            path = week2.manifest_path(root, job)
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(self.completed_manifest(job)), encoding="utf-8")
            runner = Mock(
                side_effect=subprocess.TimeoutExpired("validator", 9))
            self.write_samples(root, job)
            with patch.object(sys, "argv", [
                    "collect_week2_dataset.py", "--output", directory,
                    "--frames", "30", "--max-jobs", "1",
                    "--min-free-gb", "0", "--validator-timeout-s", "9"]), \
                    patch.object(week2, "carla_preflight", Mock()), \
                    patch.object(week2, "run_checked", runner), \
                    redirect_stdout(StringIO()):
                with self.assertRaises(subprocess.TimeoutExpired):
                    week2.main()
            self.assertEqual(runner.call_args.args[1], 9.0)
            state = json.loads(
                (root / "collection_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["completed_jobs"], [0])
            self.assertEqual(state["validation"]["status"], "failed")
            self.assertEqual(state["validation"]["timeout_s"], 9.0)


if __name__ == "__main__":
    unittest.main()
