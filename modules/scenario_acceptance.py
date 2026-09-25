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
) -> Dict[str, Any]:
    """Trả verdict + lý do thất bại có thể đọc bằng máy và bằng người.

    Every criterion here fails closed. A measurement that is missing, not a
    finite number, or physically impossible is a failure with its own reason,
    never a silent pass. Three of them used to pass: a NaN reaction delay and a
    NaN clearance (every comparison against NaN is False, so neither limit
    fired), a negative delay (a reaction before the hazard existed), and a KPI
    summary with no collision count at all (read as zero collisions). The
    2026-09-25 evidence review reproduced all three.
    """
    reasons = []

    raw_collisions = kpi_summary.get("collisions")
    collisions = _count(raw_collisions)
    min_distance = kpi_summary.get("min_distance_m")

    if not triggered:
        reasons.append("scenario không kích hoạt (actor/trigger không hợp lệ)")
    if raw_collisions is None:
        reasons.append("collision count was not observed")
    elif collisions is None:
        reasons.append(f"collision count is not a valid count: {raw_collisions!r}")
    elif collisions != 0:
        reasons.append(f"{collisions} va chạm")
    if triggered and not reacted:
        reasons.append("FSM không phản ứng sau khi scenario kích hoạt")
    if triggered and reacted:
        if reaction_delay_s is None:
            reasons.append("không đo được thời gian phản ứng")
        elif not _finite(reaction_delay_s):
            reasons.append(f"reaction delay is not a finite number: {reaction_delay_s!r}")
        elif reaction_delay_s < 0:
            reasons.append(
                f"negative reaction delay {reaction_delay_s:.3f}s - reaction timed "
                "before the hazard existed, a timestamp fault")
        elif reaction_delay_s > max_reaction_s:
            reasons.append(
                f"phản ứng trễ {reaction_delay_s:.2f}s > {max_reaction_s:.2f}s")
    # None stays "not measurable", which the criterion allows (weather-only
    # runs have no actor). A value that is present but not a finite number is
    # a broken measurement, not an absent one.
    if min_distance is not None:
        if not _finite(min_distance):
            reasons.append(f"clearance is not a finite number: {min_distance!r}")
        elif min_distance < min_clearance_m:
            reasons.append(
                f"khoảng hở tối thiểu {min_distance:.2f}m < {min_clearance_m:.2f}m")
    if sensor_frame_errors > 0:
        reasons.append(f"{sensor_frame_errors} lỗi đồng bộ frame cảm biến")
    if category == "cutin" and triggered and not (braked or evaded):
        reasons.append("cut-in không tạo hành động phanh hoặc né")

    return {
        "name": name,
        "pass": not reasons,
        "reasons": reasons,
        "criteria": {
            "scenario_triggered": True,
            "collisions": 0,
            "max_reaction_s": max_reaction_s,
            "min_clearance_m": min_clearance_m,
            "sensor_frame_errors": 0,
            "cutin_requires_brake_or_evade": category == "cutin",
        },
    }


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
