"""Ghi và chạy lại chuỗi quyết định AEB bằng Python, không cần CARLA."""

from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .control_arbitration import aeb_brake, longitudinal_override
    from .safety_contracts import SAFETY_ACTIONS, SAFETY_STATES, SafetyConfig, SafetyInput, SafetyOutput
    from .safety_fsm import SafetyFSM
else:
    from control_arbitration import aeb_brake, longitudinal_override
    from safety_contracts import SAFETY_ACTIONS, SAFETY_STATES, SafetyConfig, SafetyInput, SafetyOutput
    from safety_fsm import SafetyFSM

SCHEMA_VERSION = 1
ABS_TOL = 1e-9
EXIT_CODES = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "INVALID": 2, "ERROR": 3}


def _json(data):
    return json.dumps(data, allow_nan=False, ensure_ascii=False, sort_keys=True)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(data) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _decode(line):
    def unique_object(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate JSON key")
        return result

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number: {token}")
        return value

    return json.loads(line, parse_constant=_reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def _config_id(raw):
    return hashlib.sha256(_json(raw).encode("utf-8")).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _record(cls, raw):
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__}: missing or unexpected field")
    for key, value in raw.items():
        if key in {"lidar_valid", "left_clear", "right_clear", "enable_evasion", "collision_risk"}:
            valid = type(value) is bool
        elif key == "current_lane_id":
            valid = value is None or type(value) is int
        elif key in {"nearest_distance_m", "ttc_s"}:
            valid = value is None or _number(value)
        elif key == "state":
            valid = value in SAFETY_STATES
        elif key == "action":
            valid = value in SAFETY_ACTIONS
        else:
            valid = _number(value)
        if not valid:
            raise ValueError(f"{cls.__name__}.{key}: invalid value {value!r}")
    return cls(**raw)


def _deceleration(l3):
    # Matching float(...) at the existing Python arbitration boundary. Store null for
    # invalid/non-finite requests so replay still exercises the full-brake rule.
    try:
        value = float(l3.get("target_decel_ms2", 0.0))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _arbitrate(request, decision):
    command = longitudinal_override(request, decision)
    if command is not None:
        return command
    if decision.action == "BRAKE":
        level = aeb_brake(decision)
        return {"brake": level, "hand_brake": False, "source": "aeb",
                "aeb_brake": level, "mrm_brake": 0.0}
    return None


def _brake_command(control_status):
    mode = control_status.get("mode")
    applied = control_status.get("applied", {})
    return {
        "mode": mode,
        "command_sent": control_status.get("command_sent"),
        "brake": applied.get("brake"),
        "hand_brake": applied.get("hand_brake", False),
        "source": control_status.get("brake_source", "aeb" if mode == "aeb_override" else None),
    }


def _validate_brake(command, *, observed=False):
    if command is None and not observed:
        return
    keys = {"brake", "hand_brake", "source"}
    keys.update({"mode", "command_sent"} if observed else {"aeb_brake", "mrm_brake"})
    if not isinstance(command, dict) or command.keys() != keys:
        raise ValueError("invalid brake command fields")
    if type(command["hand_brake"]) is not bool or command["source"] not in (None, "aeb", "mrm"):
        raise ValueError("invalid brake command source or hand brake")
    if observed:
        if (command["mode"] is not None and not isinstance(command["mode"], str)
                or command["command_sent"] is not None and type(command["command_sent"]) is not bool
                or command["brake"] is not None and not _number(command["brake"])):
            raise ValueError("invalid observed brake command")
    elif not all(_number(command[key]) and 0 <= command[key] <= 1
                 for key in ("brake", "aeb_brake", "mrm_brake")):
        raise ValueError("invalid recorded brake magnitude")


class SafetyReplayWriter:
    """Streaming từng tick. Thư mục mới và manifest cuối xác nhận capture đủ."""

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema_version": SCHEMA_VERSION, "metadata": metadata, "episodes": []}
        self.stream = (self.directory / "steps.jsonl").open("x", encoding="utf-8", newline="\n")
        self.episode = None
        self.error = None
        self.closed = False

    def start_episode(self, episode_id, config):
        if self.episode is not None or self.closed:
            raise ValueError("finish the current episode before starting another")
        if any(row["episode_id"] == episode_id for row in self.manifest["episodes"]):
            raise ValueError(f"duplicate episode {episode_id}")
        raw_config = asdict(config)
        self.episode = {"episode_id": episode_id, "config": raw_config,
                        "config_id": _config_id(raw_config), "recorded_steps": 0}
        self.manifest["episodes"].append(self.episode)

    def record(self, safety, decision, l3, control_status, *, frame_id=None, phase="control",
               observations=None):
        if self.error is not None:
            return
        try:
            if self.episode is None:
                raise ValueError("start episode before the first record")
            if safety.steps != self.episode["recorded_steps"] + 1:
                raise ValueError("missing, repeated or out-of-order FSM call")
            if asdict(safety.last_config) != self.episode["config"]:
                raise ValueError("FSM configuration changed during capture")
            output = {field.name: getattr(decision, field.name) for field in fields(SafetyOutput)}
            request = {"override": bool(l3.get("override")), "state": l3.get("state"),
                       "target_decel_ms2": _deceleration(l3)}
            inputs = asdict(safety.last_input)
            _record(SafetyInput, inputs)
            _record(SafetyOutput, output)
            observed = _brake_command(control_status)
            _validate_brake(observed, observed=True)
            self.stream.write(_json({
                "episode_id": self.episode["episode_id"], "sequence_id": safety.steps,
                "frame_id": frame_id, "phase": phase, "config_id": self.episode["config_id"],
                "input": inputs, "output": output, "diagnostics": decision.threat,
                **({"observations": observations} if observations is not None else {}),
                "arbitration_input": request, "arbitration_output": _arbitrate(request, decision),
                "controller": observed,
            }) + "\n")
            self.episode["recorded_steps"] += 1
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def end_episode(self, expected_steps, *, complete):
        if self.episode is None:
            raise ValueError("no active episode")
        recorded = self.episode["recorded_steps"]
        self.episode.update(expected_steps=expected_steps, dropped_steps=max(0, expected_steps - recorded),
                            complete=complete and recorded == expected_steps and recorded > 0 and not self.error)
        self.episode = None

    def close(self):
        if self.closed:
            raise ValueError("replay already closed")
        self.closed = True
        try:
            self.stream.close()
        except OSError as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        if self.episode is not None:
            self.end_episode(self.episode["recorded_steps"], complete=False)
        episodes = self.manifest["episodes"]
        complete = (bool(episodes) and all(row["complete"] for row in episodes)
                    and not self.error)
        self.manifest.update(complete=complete, error=self.error)
        write_json(self.directory / "manifest.json", self.manifest)
        return {"directory": str(self.directory), "status": "COMPLETE" if complete else "INCOMPLETE",
                "expected_steps": sum(row["expected_steps"] for row in episodes),
                "recorded_steps": sum(row["recorded_steps"] for row in episodes),
                "dropped_steps": sum(row["dropped_steps"] for row in episodes),
                "error": self.error}


def _matches(expected, actual):
    if _number(expected) and _number(actual):
        return math.isclose(expected, actual, rel_tol=0, abs_tol=ABS_TOL)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _matches(value, actual[key]) for key, value in expected.items())
    return type(expected) is type(actual) and expected == actual


def _compare_step(fsm, config, row, report):
    inputs = _record(SafetyInput, row["input"])
    expected = asdict(_record(SafetyOutput, row["output"]))
    request = row["arbitration_input"]
    if (not isinstance(request, dict)
            or request.keys() != {"override", "state", "target_decel_ms2"}
            or type(request["override"]) is not bool
            or (request["state"] is not None and not isinstance(request["state"], str))
            or (request["target_decel_ms2"] is not None and not _number(request["target_decel_ms2"]))):
        raise ValueError("invalid arbitration request")
    output = fsm.step(inputs, config)
    actual = asdict(output)
    arbitration = _arbitrate(request, output)
    _validate_brake(row["arbitration_output"])
    differs = [name for name in expected if not _matches(expected[name], actual[name])]
    if not _matches(row["arbitration_output"], arbitration):
        differs.append("arbitration")
    # L3/AEB override is observable in the status returned by EgoController.apply.
    # Planner, Traffic Manager and fault-stop decisions are outside this comparison.
    observed = row["controller"]
    _validate_brake(observed, observed=True)
    if observed["mode"] == "custom_fault_safe_stop":
        report["controller_fault_stop_steps"] += 1
    elif arbitration is not None:
        report["controller_override_steps_compared"] += 1
        mode = "l3_override" if request["override"] else "aeb_override"
        expected_command = {"mode": mode, "command_sent": True,
                            **{key: arbitration[key] for key in ("brake", "hand_brake", "source")}}
        if not _matches(observed, expected_command):
            differs.append("controller")
    elif arbitration is None and observed["mode"] in {"l3_override", "aeb_override"}:
        differs.append("controller")
    report["compared_steps"] += 1
    for name in ("state", "action", "arbitration", "controller"):
        report[f"{name}_mismatches"] += int(name in differs)
    report["output_mismatches"] += int(bool(set(differs) & expected.keys()))
    if differs and report["first_mismatch"] is None:
        report["first_mismatch"] = {
            "episode_id": row["episode_id"], "sequence_id": row["sequence_id"],
            "frame_id": row["frame_id"], "fields": differs, "input": row["input"],
            "expected": expected, "actual": actual, "controller": observed,
            "expected_arbitration": row["arbitration_output"], "actual_arbitration": arbitration,
            "transition_reason": fsm.last_transition_reason,
        }


class IncompleteReplay(ValueError):
    """Capture chưa đóng hoặc thiếu tick."""


def compare_replay(directory):
    directory = Path(directory)
    report = {"schema_version": SCHEMA_VERSION, "status": "INCOMPLETE", "reason": None,
              "absolute_tolerance": ABS_TOL, "expected_steps": None, "recorded_steps": None,
              "compared_steps": 0, "dropped_steps": None, "state_mismatches": 0,
              "action_mismatches": 0, "output_mismatches": 0, "arbitration_mismatches": 0,
              "controller_mismatches": 0, "controller_override_steps_compared": 0,
              "controller_fault_stop_steps": 0, "first_mismatch": None,
              "first_mismatch_sequence_id": None}
    try:
        manifest = _decode((directory / "manifest.json").read_text(encoding="utf-8"))
        if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION
                or type(manifest["complete"]) is not bool):
            raise ValueError("unsupported manifest schema")
        episodes = manifest["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise IncompleteReplay("no episodes")
        configs = {}
        counts = {}
        ids = set()
        for episode in episodes:
            episode_id = episode["episode_id"]
            if not isinstance(episode_id, str) or not episode_id or episode_id in ids:
                raise ValueError("invalid or repeated episode ID")
            ids.add(episode_id)
            for counter in ("expected_steps", "recorded_steps", "dropped_steps"):
                if not _count(episode[counter]):
                    raise ValueError("invalid episode counter")
            if type(episode["complete"]) is not bool or episode["config_id"] != _config_id(episode["config"]):
                raise ValueError("invalid episode completion flag or configuration hash")
            configs[episode_id] = _record(SafetyConfig, episode["config"])
            counts[episode_id] = {"episode_id": episode_id, "expected_steps": episode["expected_steps"],
                                  "recorded_steps": 0, "compared_steps": 0,
                                  "config_id": episode["config_id"]}
        report["episodes"] = list(counts.values())
        report["expected_steps"] = sum(ep["expected_steps"] for ep in episodes)
        report["metadata"] = manifest["metadata"]
        if not isinstance(report["metadata"], dict):
            raise ValueError("invalid capture metadata")
        incomplete = manifest.get("error") or (None if manifest["complete"] else "capture was incomplete")
        order = {ep["episode_id"]: index for index, ep in enumerate(episodes)}
        current = -1
        with (directory / "steps.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    incomplete = "truncated step"
                    break
                row = _decode(line)
                episode_id = row["episode_id"]
                index = order[episode_id]
                ep = counts[episode_id]
                if index < current:
                    raise ValueError("out-of-order episode")
                if index != current:
                    fsm = SafetyFSM()
                    current = index
                    last_sequence = 0
                seq = row["sequence_id"]
                if not _count(seq) or not last_sequence < seq <= ep["expected_steps"]:
                    raise ValueError(f"invalid or repeated sequence in {episode_id}: {seq!r}")
                if seq != last_sequence + 1:
                    incomplete = incomplete or f"missing {episode_id} step {last_sequence + 1}"
                last_sequence = seq
                if row["config_id"] != ep["config_id"]:
                    raise ValueError("step references a different configuration")
                if row["phase"] not in {"warmup", "control"} or not isinstance(row["diagnostics"], dict):
                    raise ValueError("invalid capture phase or diagnostics")
                if row["frame_id"] is not None and not _count(row["frame_id"]):
                    raise ValueError("invalid world frame")
                ep["recorded_steps"] += 1
                # After a gap, state history is unknown until the next episode reset.
                if seq == ep["compared_steps"] + 1:
                    _compare_step(fsm, configs[episode_id], row, report)
                    ep["compared_steps"] += 1
        for declared in episodes:
            ep = counts[declared["episode_id"]]
            ep["dropped_steps"] = ep["expected_steps"] - ep["recorded_steps"]
            if (not declared["complete"] or not ep["expected_steps"] or ep["dropped_steps"]
                    or declared["dropped_steps"] or ep["recorded_steps"] != declared["recorded_steps"]):
                incomplete = incomplete or f"incomplete episode {ep['episode_id']}"
        report["recorded_steps"] = sum(ep["recorded_steps"] for ep in counts.values())
        report["dropped_steps"] = sum(ep["dropped_steps"] for ep in counts.values())
        if incomplete:
            raise IncompleteReplay(incomplete)
        report["status"] = "FAIL" if report["first_mismatch"] else "PASS"
    except (FileNotFoundError, IncompleteReplay) as exc:
        report.update(status="INCOMPLETE", reason=str(exc))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        report.update(status="INVALID", reason=str(exc))
    except OSError as exc:
        report.update(status="ERROR", reason=str(exc))
    if report["first_mismatch"] is not None:
        report["first_mismatch_sequence_id"] = report["first_mismatch"]["sequence_id"]
    return report


def finish_capture(writer, expected_steps, complete):
    """Đóng capture và báo trạng thái độc lập với kết quả điều khiển xe."""
    try:
        writer.end_episode(expected_steps, complete=complete)
        return writer.close()
    except (OSError, ValueError) as exc:
        return {"status": "ERROR", "directory": str(writer.directory),
                "reason": f"{type(exc).__name__}: {exc}"}
