"""Tiêu chí nghiệm thu định lượng cho scenario ADAS.

Module thuần Python để CI có thể kiểm tra mà không cần CARLA. Một scenario chỉ
được xem là hợp lệ khi nó thực sự kích hoạt; sau đó ego phải phản ứng đúng hạn,
không va chạm và không có lỗi đồng bộ cảm biến.
"""

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
    """Trả verdict + lý do thất bại có thể đọc bằng máy và bằng người."""
    reasons = []

    collisions = int(kpi_summary.get("collisions", 0))
    min_distance = kpi_summary.get("min_distance_m")

    if not triggered:
        reasons.append("scenario không kích hoạt (actor/trigger không hợp lệ)")
    if collisions != 0:
        reasons.append(f"{collisions} va chạm")
    if triggered and not reacted:
        reasons.append("FSM không phản ứng sau khi scenario kích hoạt")
    if triggered and reacted:
        if reaction_delay_s is None:
            reasons.append("không đo được thời gian phản ứng")
        elif reaction_delay_s > max_reaction_s:
            reasons.append(
                f"phản ứng trễ {reaction_delay_s:.2f}s > {max_reaction_s:.2f}s")
    if min_distance is not None and min_distance < min_clearance_m:
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
