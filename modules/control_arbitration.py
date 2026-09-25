"""Longitudinal arbitration between an active L3 MRM and the AEB.

Pure Python, no CARLA import, so the rule can be tested in CI.

WHY THIS EXISTS
---------------
An MRM is a *comfort* stop: it decelerates at a fraction of available friction
so a driver who failed to take over is brought to rest without being thrown
into the belt. The AEB is the opposite: it brakes as hard as the situation
needs because something is in the path.

The two used to be ranked, MRM above AEB, and the lower-ranked request was
dropped. So an MRM in progress - typically 1-2 m/s^2 - silently discarded an
AEB request for full braking. The 2026-09-25 evidence review reproduced it:
AEB asked for 1.0, the vehicle received 0.167.

Ranking is the wrong model for two brake requests. Both are requests to slow
down, and the correct combination of "slow down gently" and "stop now" is
"stop now". This module returns the stronger of the two. The MRM keeps
everything else it owns - hazard lights, the SAFE_STOP hand brake, and the
fact that it, not the planner, is in charge.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping, Optional

#: The MRM's requested deceleration (m/s^2) that maps to a full brake pedal.
#: Unchanged from the value previously inlined in ego_control.
MRM_DECEL_AT_FULL_BRAKE = 6.0


def _unit(value: Any) -> float:
    """Clamp a brake request into [0, 1]. A request that is not a finite number
    is treated as a request for full braking: an invalid safety command must
    not decay into no command."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return min(1.0, max(0.0, v))


def longitudinal_override(l3: Mapping[str, Any], decision: Any) -> Optional[dict]:
    """Brake command while the L3 layer overrides the planner, else None.

    Returns ``{"brake", "hand_brake", "source"}`` where ``source`` is ``"mrm"``
    or ``"aeb"`` depending on which request set the pedal.
    """
    if not l3.get("override"):
        return None

    state = l3.get("state")
    if state == "SAFE_STOP":
        mrm_brake = 1.0
    else:
        decel = l3.get("target_decel_ms2", 0.0)
        try:
            decel_f = float(decel)
        except (TypeError, ValueError):
            decel_f = math.nan
        mrm_brake = (_unit(decel_f / MRM_DECEL_AT_FULL_BRAKE)
                     if math.isfinite(decel_f) else 1.0)

    aeb_brake = 0.0
    if getattr(decision, "action", None) == "BRAKE":
        aeb_brake = _unit(getattr(decision, "brake", 1.0))

    return {
        "brake": max(mrm_brake, aeb_brake),
        "hand_brake": state == "SAFE_STOP",
        "source": "aeb" if aeb_brake > mrm_brake else "mrm",
        # Both requests are kept so a trace can show what was asked for, not
        # only what was selected.
        "aeb_brake": aeb_brake,
        "mrm_brake": mrm_brake,
    }


def aeb_brake(decision: Any) -> float:
    """The AEB brake request on the [0, 1] pedal scale, validated."""
    return _unit(getattr(decision, "brake", 1.0))


_LIMITS = {"throttle": (0.0, 1.0), "brake": (0.0, 1.0), "steer": (-1.0, 1.0)}


def command_problems(**fields: Any) -> list:
    """Why a control command must not reach the simulator, or [] if it may.

    A bool is rejected even though Python treats it as an int: ``True`` is
    not a measured pedal position. Nothing is clamped here - clamping a NaN
    or a 7.0 into range hides the fault that produced it. The caller decides
    what to do instead (the custom stack treats it as a control fault and
    safe-stops).
    """
    problems = []
    for name, value in fields.items():
        low, high = _LIMITS[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            problems.append(f"{name} is not a number: {value!r}")
        elif not math.isfinite(float(value)):
            problems.append(f"{name} is not finite: {value!r}")
        elif not low <= value <= high:
            problems.append(f"{name} {value!r} outside [{low}, {high}]")
    return problems


def exclusive_pedals(throttle: float, brake: float) -> float:
    """Throttle to send given a brake: zero whenever the brake is applied.

    Pressing both pedals is never a valid command; the brake wins.
    """
    return 0.0 if brake > 0.0 else throttle
