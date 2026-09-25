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
    }
