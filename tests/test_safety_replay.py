"""Regression theo tick cho lõi Python và hợp đồng replay AEB."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from modules.active_safety import ActiveSafetySystem
from modules.control_arbitration import longitudinal_override
from modules.safety_contracts import SafetyConfig, SafetyInput
from modules.safety_fsm import SafetyFSM
from modules.safety_replay import SafetyReplayWriter, compare_replay

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "safety"
BASELINE = json.loads((FIXTURES / "python_baseline.json").read_text(encoding="utf-8"))
BRAKE = [{"centroid": [4, 0, 0], "point_count": 20}]
L3 = {"override": False, "state": "L3_ACTIVE"}


def _observation(decision, l3):
    command = longitudinal_override(l3, decision)
    if command is not None:
        return {"mode": "l3_override", "command_sent": True,
                "brake_source": command["source"], "applied": command}
    if decision.action == "BRAKE":
        return {"mode": "aeb_override", "command_sent": True,
                "applied": {"brake": decision.brake}}
    return {"mode": "custom"}


class SafetyBaselineTests(unittest.TestCase):
    def test_original_update_matches_after_refactor_and_reset(self):
        safety = ActiveSafetySystem()
        for episode in BASELINE["episodes"]:
            safety.reset()
            safety.set_conditions(0.8)
            for index, row in enumerate(episode["steps"], 1):
                with self.subTest(episode=episode["id"], sequence=index):
                    data = dict(row["input"])
                    if "conditions" in data:
                        safety.set_conditions(*data.pop("conditions"))
                    if "tracks" in data:
                        data["tracks"] = [SimpleNamespace(**track) for track in data["tracks"]]
                    self.assertEqual(asdict(safety.update(**data)), row["decision"])
                    self.assertEqual(safety.last_transition_reason, row["transition_reason"])
                    self.assertEqual(safety.steps, index)

    def test_original_arbitration(self):
        for case in BASELINE["arbitration"]:
            decision = SimpleNamespace(action=case["action"], brake=case["brake"])
            self.assertEqual(longitudinal_override(case["l3"], decision), case["expected"])


class SafetyContractTests(unittest.TestCase):
    def setUp(self):
        self.config = SafetyConfig()
        self.inputs = SafetyInput(0.025, 12.0, 15.0, None, 0.0, True, True, True, 1, 0.8, 1.0)

    def test_snapshot_is_immutable_and_keeps_effective_conditions(self):
        safety = ActiveSafetySystem()
        safety.set_conditions(0.4, 1.3)
        safety.update(12, [], BRAKE, 0.025)
        safety.set_conditions(0.8)
        self.assertEqual((safety.last_input.mu, safety.last_input.gap_multiplier), (0.4, 1.3))
        self.assertEqual(safety.last_config, safety.fsm_config())
        with self.assertRaises(FrozenInstanceError):
            safety.last_input.mu = 1.0

    def test_adjacent_floats_at_distance_and_ttc_boundaries(self):
        for key, threshold in (("nearest_distance_m", 6.0), ("ttc_s", 1.6)):
            for value, state in ((math.nextafter(threshold, 0.0), "EMERGENCY_BRAKE"),
                                 (threshold, "FOLLOW"),
                                 (math.nextafter(threshold, math.inf), "FOLLOW")):
                with self.subTest(key=key, value=value):
                    outcome = SafetyFSM().step(replace(self.inputs, **{key: value}), self.config)
                    self.assertEqual(outcome.state, state)

    def test_missing_lane_id_does_not_complete_committed_evasion(self):
        fsm = SafetyFSM()
        step = replace(self.inputs, nearest_distance_m=20.0,
                       conservative_closing_ms=11.0, ttc_s=2.5)
        self.assertEqual(fsm.step(step, self.config).state, "EVADE_RIGHT")
        self.assertEqual(fsm.step(replace(step, current_lane_id=None), self.config).state, "EVADE_RIGHT")
        blocked = fsm.step(replace(step, right_clear=False), self.config)
        self.assertEqual((blocked.state, blocked.brake), ("EMERGENCY_BRAKE", 1.0))
        fsm.reset()
        self.assertEqual(fsm.step(replace(step, nearest_distance_m=None), self.config).action, "DRIVE")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "capture"
        self.rows_path = self.directory / "steps.jsonl"
        self.metadata = {"source": "synthetic", "baseline_commit": BASELINE["source_commit"]}

    def capture(self):
        writer = SafetyReplayWriter(self.directory, self.metadata)
        a = ActiveSafetySystem()
        writer.start_episode("brake", a.fsm_config())
        for index, obstacles in enumerate([[], BRAKE, [], [], [], [], [], []], 1):
            decision = a.update(10, [], obstacles, 0.025, lidar_valid=index != 3)
            l3 = ({"override": True, "state": "MRM_EXECUTING", "target_decel_ms2": 1.0}
                  if index == 2 else L3)
            writer.record(a, decision, l3, _observation(decision, l3), frame_id=index * 2,
                          phase="warmup" if index == 1 else "control")
        writer.end_episode(8, complete=True)
        a.reset()
        writer.start_episode("reset", a.fsm_config())
        decision = a.update(10, [], [], 0.025)
        l3 = {"override": True, "state": "SAFE_STOP", "target_decel_ms2": math.nan}
        writer.record(a, decision, l3, _observation(decision, l3))
        writer.end_episode(1, complete=True)
        self.assertEqual(writer.close()["status"], "COMPLETE")

    def rows(self):
        return [json.loads(line) for line in self.rows_path.read_text(encoding="utf-8").splitlines()]

    def write_rows(self, rows):
        self.rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_roundtrip_and_episode_reset(self):
        self.capture()
        report = compare_replay(self.directory)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual([report[name] for name in ("expected_steps", "recorded_steps", "compared_steps",
                                                    "dropped_steps")], [9, 9, 9, 0])
        rows = self.rows()
        self.assertEqual(rows[0]["phase"], "warmup")
        self.assertEqual(rows[0]["output"]["action"], "DRIVE")
        self.assertIsNone(rows[0]["input"]["ttc_s"])
        self.assertEqual(rows[2]["input"]["lidar_valid"], False)

    def test_mismatch_reports_first_bad_step_and_cli_exit(self):
        self.capture()
        rows = self.rows()
        rows[1]["output"]["brake"] = 0.2
        self.write_rows(rows)
        report = compare_replay(self.directory)
        self.assertEqual((report["status"], report["first_mismatch_sequence_id"]), ("FAIL", 2))
        self.assertEqual(report["output_mismatches"], 1)
        target = self.directory / "comparison.json"
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/tools/compare_safety_replay.py"),
             "--input", str(self.directory), "--report", str(target)],
            cwd=self.directory.parent, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(target.is_file())
        self.assertTrue((self.directory / "first_mismatch.json").is_file())

    def test_truncated_or_missing_tick_is_incomplete(self):
        self.capture()
        raw = self.rows_path.read_text(encoding="utf-8")
        self.rows_path.write_text(raw[:-7], encoding="utf-8")
        report = compare_replay(self.directory)
        self.assertEqual((report["status"], report["recorded_steps"], report["dropped_steps"]),
                         ("INCOMPLETE", 8, 1))
        self.rows_path.write_text(raw, encoding="utf-8")
        rows = self.rows()
        del rows[2]
        self.write_rows(rows)
        report = compare_replay(self.directory)
        self.assertEqual((report["status"], report["recorded_steps"], report["dropped_steps"]),
                         ("INCOMPLETE", 8, 1))
        self.assertEqual([ep["compared_steps"] for ep in report["episodes"]], [2, 1])

    def test_missing_diagnostics_and_conflicting_writer_error_never_pass(self):
        self.capture()
        rows = self.rows()
        del rows[0]["diagnostics"]
        self.write_rows(rows)
        self.assertEqual(compare_replay(self.directory)["status"], "INVALID")
        rows[0]["diagnostics"] = {}
        self.write_rows(rows)
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["error"] = "flush failed"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(compare_replay(self.directory)["status"], "INCOMPLETE")

    def test_invalid_input_and_wrong_config_can_never_pass(self):
        self.capture()
        baseline = self.rows()
        for value in (math.nan, True, "0.025"):
            with self.subTest(value=value):
                rows = deepcopy(baseline)
                rows[1]["input"]["dt_s"] = value
                self.write_rows(rows)
                self.assertEqual(compare_replay(self.directory)["status"], "INVALID")
        rows = deepcopy(baseline)
        del rows[1]["input"]["dt_s"]
        self.write_rows(rows)
        self.assertEqual(compare_replay(self.directory)["status"], "INVALID")
        rows = baseline
        rows[0]["config_id"] = "bad-config"
        self.write_rows(rows)
        self.assertEqual(compare_replay(self.directory)["status"], "INVALID")

    def test_writer_reports_a_dropped_step(self):
        writer = SafetyReplayWriter(self.directory, self.metadata)
        safety = ActiveSafetySystem()
        writer.start_episode("dropped", safety.fsm_config())
        safety.update(10, [], BRAKE, 0.025)
        output = safety.update(10, [], [], 0.025)
        writer.record(safety, output, L3, _observation(output, L3))
        writer.end_episode(2, complete=True)
        self.assertEqual(writer.close()["dropped_steps"], 2)
        self.assertEqual(compare_replay(self.directory)["status"], "INCOMPLETE")

    def test_bundled_capture(self):
        report = compare_replay(FIXTURES / "python_replay")
        self.assertEqual(report["status"], "PASS", report)
        self.assertGreater(report["compared_steps"], 0)


if __name__ == "__main__":
    unittest.main()
