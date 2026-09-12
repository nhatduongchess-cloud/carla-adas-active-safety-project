"""Machine-readable acceptance for expected ODD behavior by weather profile."""


VALID_ODD_STATES = {"NORMAL", "DEGRADED", "VIOLATION"}


def assess_weather_behavior(expected_odd, observed_states, mrm_seen=False,
                            external_violation_expected=False):
    """Assess weather behavior without masking an independently proven fault.

    ``external_violation_expected`` is only valid when another acceptance layer
    positively identifies a separate reason for ODD VIOLATION (currently a
    confirmed LiDAR+radar redundancy loss).  The expected weather state must
    still be observed; this flag only avoids mislabeling the known fault state
    as a false *weather* violation.
    """
    expected = str(expected_odd or "").upper()
    observed = {str(state).upper() for state in (observed_states or ())}
    reasons = []
    if expected not in VALID_ODD_STATES:
        return {"pass": True, "reasons": [], "expected": None,
                "observed": sorted(observed), "mrm_required": False}
    if expected not in observed:
        reasons.append(f"expected ODD {expected}, observed {sorted(observed)}")
    if (expected != "VIOLATION" and "VIOLATION" in observed
            and not external_violation_expected):
        reasons.append(f"false ODD VIOLATION in {expected} profile")
    if expected == "VIOLATION" and not mrm_seen:
        reasons.append("ODD VIOLATION did not trigger MRM/safe stop")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "expected": expected,
        "observed": sorted(observed),
        "mrm_required": expected == "VIOLATION",
        "external_violation_expected": bool(external_violation_expected),
    }
