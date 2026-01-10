from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HoldControlOutput:
    roll: float
    pitch: float
    used_quality: float
    vx_px_s: float
    vy_px_s: float


class VelocityHoldController:
    def __init__(
        self,
        *,
        kp_vx: float = 0.003,
        kp_vy: float = 0.003,
        ki_vx: float = 0.0005,
        ki_vy: float = 0.0005,
        integrator_limit: float = 0.35,
        deadband_px_s: float = 3.0,
        max_cmd: float = 0.35,
        min_quality: float = 0.15,
        roll_sign: float = -1.0,
        pitch_sign: float = -1.0,
    ) -> None:
        self._kp_vx = float(kp_vx)
        self._kp_vy = float(kp_vy)
        self._ki_vx = float(ki_vx)
        self._ki_vy = float(ki_vy)
        self._integrator_limit = float(integrator_limit)
        self._deadband = float(deadband_px_s)
        self._max_cmd = float(max_cmd)
        self._min_quality = float(min_quality)
        self._roll_sign = float(roll_sign)
        self._pitch_sign = float(pitch_sign)

        self.reset()

    def reset(self) -> None:
        self._ivx = 0.0
        self._ivy = 0.0

    def update(
        self,
        *,
        dt_sec: float,
        vx_px_s: float,
        vy_px_s: float,
        quality: float,
    ) -> Optional[HoldControlOutput]:
        q = float(quality)
        if q < self._min_quality:
            self.reset()
            return None

        dt = float(dt_sec)
        if dt <= 1e-6:
            return None

        vx = 0.0 if abs(float(vx_px_s)) < self._deadband else float(vx_px_s)
        vy = 0.0 if abs(float(vy_px_s)) < self._deadband else float(vy_px_s)

        self._ivx = self._clamp(self._ivx + (vx * dt), -self._integrator_limit, self._integrator_limit)
        self._ivy = self._clamp(self._ivy + (vy * dt), -self._integrator_limit, self._integrator_limit)

        u_roll = (self._kp_vx * vx) + (self._ki_vx * self._ivx)
        u_pitch = (self._kp_vy * vy) + (self._ki_vy * self._ivy)

        roll = self._clamp(self._roll_sign * u_roll, -self._max_cmd, self._max_cmd)
        pitch = self._clamp(self._pitch_sign * u_pitch, -self._max_cmd, self._max_cmd)

        return HoldControlOutput(
            roll=float(roll),
            pitch=float(pitch),
            used_quality=float(q),
            vx_px_s=float(vx_px_s),
            vy_px_s=float(vy_px_s),
        )

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(min(hi, max(lo, v)))
