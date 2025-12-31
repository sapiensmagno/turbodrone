from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionedVelocity:
    vx_px_s: float
    vy_px_s: float
    gated: bool


class VelocityMeasurementConditioner:
    def __init__(self, *, deadband_px_s: float) -> None:
        self._deadband = float(deadband_px_s)

    def apply(self, *, vx_px_s: float, vy_px_s: float) -> ConditionedVelocity:
        vx0 = float(vx_px_s)
        vy0 = float(vy_px_s)

        vx = 0.0 if abs(vx0) < self._deadband else vx0
        vy = 0.0 if abs(vy0) < self._deadband else vy0

        return ConditionedVelocity(vx_px_s=float(vx), vy_px_s=float(vy), gated=bool(vx != vx0 or vy != vy0))
