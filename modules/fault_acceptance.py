"""Acceptance checks for injected sensor failures.

The scenario verdict validates driving behavior.  These checks validate that
the requested fault was actually observed and that degradation followed the
range-redundancy policy, including transient faults that recover before the
scenario report is written.
"""


def assess_fault_behavior(fault: str, sensor_health: dict,
                          odd_violation_seen: bool = False,
                          mrm_seen: bool = False) -> dict:
    fault = str(fault or "none")
    availability = sensor_health.get("availability", {})
    ever_range_lost = bool(sensor_health.get("range_redundancy_ever_lost", False))
    reasons = []

    expected_missing = {
        "camera-loss": ("camera",),
        "radar-loss": ("radar",),
        "lidar-loss": ("lidar",),
        "lidar-radar-loss": ("lidar", "radar"),
    }.get(fault, ())
    for sensor in expected_missing:
        if float(availability.get(sensor, 1.0)) >= 1.0:
            reasons.append(f"fault {fault} was not observed by {sensor}")

    if fault in ("camera-loss", "radar-loss", "lidar-loss") and ever_range_lost:
        reasons.append("single-sensor fault incorrectly lost range redundancy")
    if fault == "lidar-radar-loss":
        if not ever_range_lost:
            reasons.append("dual range fault did not trigger redundancy loss")
        if not odd_violation_seen:
            reasons.append("dual range fault did not trigger ODD VIOLATION")
        if not mrm_seen:
            reasons.append("dual range fault did not trigger MRM/safe stop")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "criteria": {
            "fault_observed": list(expected_missing),
            "single_sensor_preserves_range_redundancy": fault != "lidar-radar-loss",
            "dual_range_fault_requires_odd_mrm": fault == "lidar-radar-loss",
        },
    }
