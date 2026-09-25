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
    enabled = sensor_health.get("enabled", {})
    ever_range_lost = bool(sensor_health.get("range_redundancy_ever_lost", False))
    reasons = []

    expected_missing = {
        "camera-loss": ("camera",),
        "radar-loss": ("radar",),
        "lidar-loss": ("lidar",),
        "lidar-radar-loss": ("lidar", "radar"),
    }.get(fault, ())
    for sensor in expected_missing:
        if enabled.get(sensor, True) is False:
            # A fault injected into a disabled sensor tests nothing.
            reasons.append(f"fault {fault} targets {sensor}, which is disabled")
            continue
        value = availability.get(sensor)
        if value is None or float(value) >= 1.0:
            reasons.append(f"fault {fault} was not observed by {sensor}")

    # Which range sensors are left once the injected fault has removed its
    # targets. Losing the last one is total loss of range sensing and must
    # escalate exactly like the dual fault - with radar disabled, a lidar-loss
    # fault is that case, and was previously marked a failure for escalating.
    range_enabled = [name for name in ("lidar", "radar") if enabled.get(name, True)]
    survivors = [name for name in range_enabled if name not in expected_missing]
    expects_range_loss = bool(expected_missing) and not survivors and any(
        name in expected_missing for name in range_enabled)

    if fault in ("camera-loss", "radar-loss", "lidar-loss") and ever_range_lost \
            and not expects_range_loss:
        reasons.append("single-sensor fault incorrectly lost range redundancy")
    if expects_range_loss:
        if not ever_range_lost:
            reasons.append("total range fault did not trigger range loss")
        if not odd_violation_seen:
            reasons.append("total range fault did not trigger ODD VIOLATION")
        if not mrm_seen:
            reasons.append("total range fault did not trigger MRM/safe stop")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "criteria": {
            "fault_observed": list(expected_missing),
            "single_sensor_preserves_range_redundancy": not expects_range_loss,
            "dual_range_fault_requires_odd_mrm": expects_range_loss,
        },
    }
