"""Tiêu chí nghiệm thu định lượng cho scenario ADAS.

Module thuần Python để CI có thể kiểm tra mà không cần CARLA. Một scenario chỉ
được xem là hợp lệ khi nó thực sự kích hoạt; sau đó ego phải phản ứng đúng hạn,
không va chạm và không có lỗi đồng bộ cảm biến.
"""

import math
from typing import Any, Dict, Optional


DEFAULT_MAX_REACTION_S = 1.0
DEFAULT_MIN_CLEARANCE_M = 0.25


def assess_scenario(
    name: str,
    category: str,
    kpi_summary: Dict[str, Any],
    *,
    triggered: bool,
    reacted: bool,
    reaction_delay_s: Optional[float],
    braked: bool,
    evaded: bool,
    sensor_frame_errors: int = 0,
    max_reaction_s: float = DEFAULT_MAX_REACTION_S,
    min_clearance_m: float = DEFAULT_MIN_CLEARANCE_M,
    scenario_errors: Optional[list] = None,
) -> Dict[str, Any]:
    """Trả verdict + lý do thất bại có thể đọc bằng máy và bằng người.

    Every criterion here fails closed. A measurement that is missing, not a
    finite number, or physically impossible is a failure with its own reason,
    never a silent pass. Three of them used to pass: a NaN reaction delay and a
    NaN clearance (every comparison against NaN is False, so neither limit
    fired), a negative delay (a reaction before the hazard existed), and a KPI
    summary with no collision count at all (read as zero collisions). The
    2026-09-25 evidence review reproduced all three.

    ``status`` is PASS, FAIL (a requirement observably not met) or INVALID (the
    case cannot be judged). ``pass`` is kept for existing consumers and is
    True only for PASS.
    """
    _check_threshold("max_reaction_s", max_reaction_s)
    _check_threshold("min_clearance_m", min_clearance_m)
    # Two kinds of reason, kept apart. A FAILURE is evidence that a requirement
    # was not met. An INVALID reason means the case cannot be judged (setup,
    # oracle or measurement broken). Both are kept when both occur; an observed
    # failure is never hidden behind "invalid".
    failures, invalid = [], []

    raw_collisions = kpi_summary.get("collisions")
    collisions = _count(raw_collisions)
    min_distance = kpi_summary.get("min_distance_m")
    frame_errors = _count(sensor_frame_errors)

    if not triggered:
        invalid.append("scenario không kích hoạt (actor/trigger không hợp lệ)")
    for error in scenario_errors or ():
        invalid.append(f"scenario execution error: {error}")
    if raw_collisions is None:
        invalid.append("collision count was not observed")
    elif collisions is None:
        invalid.append(f"collision count is not a valid count: {raw_collisions!r}")
    elif collisions != 0:
        failures.append(f"{collisions} va chạm")
    if triggered and not reacted:
        failures.append("FSM không phản ứng sau khi scenario kích hoạt")
    if triggered and reacted:
        if reaction_delay_s is None:
            invalid.append("không đo được thời gian phản ứng")
        elif not _finite(reaction_delay_s):
            invalid.append(f"reaction delay is not a finite number: {reaction_delay_s!r}")
        elif reaction_delay_s < 0:
            invalid.append(
                f"negative reaction delay {reaction_delay_s:.3f}s - reaction timed "
                "before the hazard existed, a timestamp fault")
        elif reaction_delay_s > max_reaction_s:
            failures.append(
                f"phản ứng trễ {reaction_delay_s:.2f}s > {max_reaction_s:.2f}s")
    # None stays "not measurable", which the criterion allows (weather-only
    # runs have no actor). A value that is present but not a finite number is
    # a broken measurement, not an absent one.
    if min_distance is not None:
        if not _finite(min_distance):
            invalid.append(f"clearance is not a finite number: {min_distance!r}")
        elif min_distance < min_clearance_m:
            failures.append(
                f"khoảng hở tối thiểu {min_distance:.2f}m < {min_clearance_m:.2f}m")
    if frame_errors is None:
        invalid.append(f"sensor frame error count is not a valid count: {sensor_frame_errors!r}")
    elif frame_errors > 0:
        failures.append(f"{frame_errors} lỗi đồng bộ frame cảm biến")
    if category == "cutin" and triggered and not (braked or evaded):
        failures.append("cut-in không tạo hành động phanh hoặc né")

    reasons = failures + invalid
    status = "FAIL" if failures else "INVALID" if invalid else "PASS"
    return {
        "name": name,
        "status": status,
        "pass": status == "PASS",
        "reasons": reasons,
        "failure_reasons": failures,
        "invalid_reasons": invalid,
        "criteria": {
            "scenario_triggered": True,
            "collisions": 0,
            "max_reaction_s": max_reaction_s,
            "min_clearance_m": min_clearance_m,
            "sensor_frame_errors": 0,
            "cutin_requires_brake_or_evade": category == "cutin",
        },
    }


#: Version of the reaction-latency definition. v1 (all published reports up to
#: 2026-09-25) took the first non-NORMAL state after the TRIGGER and clamped
#: ``reaction - hazard`` at zero, so a reaction that happened and ended before
#: the hazard existed was scored as an instant 0 s response. v2 is below.
REACTION_METRIC_VERSION = 2


def reaction_timing(hazard_frame, first_reaction_frame, first_reaction_after_hazard_frame, dt):
    """Reaction latency, v2.

    * No hazard: no latency (the case is INVALID for "not triggered").
    * The latency is measured to the first non-NORMAL safety state at or after
      the hazard frame - never clamped from a negative number.
    * A reaction that started before the hazard and was still active on the
      hazard frame is a *preemptive response*: latency 0.0, flagged.
    * A reaction that started and ended before the hazard is not a response
      to it: ``reacted_to_hazard`` is False.

    What it still does not do: tie the reaction to the hazard actor. A FOLLOW
    state for some other object that happens to be active counts. It measures
    the safety DECISION, not the applied command or the vehicle's motion.
    """
    result = {
        "reaction_metric_version": REACTION_METRIC_VERSION,
        "reaction_delay_s": None,
        "reacted_to_hazard": False,
        "preemptive_response": False,
        "first_reaction_minus_hazard_s": None,
    }
    if hazard_frame is None:
        return result
    if first_reaction_frame is not None:
        result["first_reaction_minus_hazard_s"] = round(
            (first_reaction_frame - hazard_frame) * dt, 6)
    if first_reaction_after_hazard_frame is None:
        return result
    result["reacted_to_hazard"] = True
    if (first_reaction_frame is not None and first_reaction_frame < hazard_frame
            and first_reaction_after_hazard_frame == hazard_frame):
        result["preemptive_response"] = True
    result["reaction_delay_s"] = (first_reaction_after_hazard_frame - hazard_frame) * dt
    return result


def finalize_after_cleanup(row: Dict[str, Any], cleanup_errors: list) -> Dict[str, Any]:
    """Attach cleanup evidence to a case row and let it change the verdict.

    A case whose teardown failed is not a clean PASS: the next case may inherit
    its actors, and "passed, then broke the world" is not a result to publish.
    An observed FAIL stays FAIL (with the cleanup reason added); a PASS
    becomes ERROR.
    """
    errors = list(cleanup_errors or [])
    row["cleanup"] = {"verified": not errors, "errors": errors}
    if errors:
        row["pass"] = False
        row["reasons"] = list(row.get("reasons") or []) + [
            "cleanup failed: " + "; ".join(errors)]
        if row.get("status") in (None, "PASS"):
            row["status"] = "ERROR"
    return row


def _check_threshold(name: str, value: Any) -> None:
    """An acceptance threshold that is not a finite positive number is a
    configuration error; fail before anything is judged against it."""
    if not _finite(value) or float(value) <= 0:
        raise ValueError(f"acceptance threshold {name} must be a finite positive number, "
                         f"got {value!r}")


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count(value: Any) -> Optional[int]:
    """A non-negative whole number, or None if the value is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0 or f != int(f):
        return None
    return int(f)
