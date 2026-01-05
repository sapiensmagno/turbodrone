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


@dataclass(frozen=True)
class PositionLeashState:
    pos_x: float
    pos_y: float
    err_x: float
    err_y: float
    v_sp_x: float
    v_sp_y: float


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


class PositionLeashController:
    def __init__(
        self,
        *,
        inner: VelocityHoldController,
        kp_pos: float,
        ki_pos: float,
        pos_deadband: float,
        max_v_sp: float,
        pos_integrator_limit: float,
        min_quality: float,
    ) -> None:
        self._inner = inner
        self._kp_pos = float(kp_pos)
        self._ki_pos = float(ki_pos)
        self._pos_deadband = float(pos_deadband)
        self._max_v_sp = float(max_v_sp)
        self._pos_integrator_limit = float(pos_integrator_limit)
        self._min_quality = float(min_quality)

        self.reset()

    def reset(self) -> None:
        self._inner.reset()
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._ipx = 0.0
        self._ipy = 0.0
        self._state = PositionLeashState(pos_x=0.0, pos_y=0.0, err_x=0.0, err_y=0.0, v_sp_x=0.0, v_sp_y=0.0)

    def state(self) -> PositionLeashState:
        return self._state

    def update(
        self,
        *,
        dt_sec: float,
        vx_px_s: float,
        vy_px_s: float,
        pos_x: Optional[float] = None,
        pos_y: Optional[float] = None,
        quality: float,
    ) -> Optional[HoldControlOutput]:
        q = float(quality)
        if q < self._min_quality:
            self.reset()
            return None

        dt = float(dt_sec)
        if dt <= 1e-6:
            return None

        vx = float(vx_px_s)
        vy = float(vy_px_s)

        if pos_x is None or pos_y is None:
            self._pos_x = float(self._pos_x + (vx * dt))
            self._pos_y = float(self._pos_y + (vy * dt))
        else:
            self._pos_x = float(pos_x)
            self._pos_y = float(pos_y)

        err_x = float(self._pos_x)
        err_y = float(self._pos_y)
        if abs(err_x) < self._pos_deadband:
            err_x = 0.0
        if abs(err_y) < self._pos_deadband:
            err_y = 0.0

        self._ipx = self._clamp(self._ipx + (err_x * dt), -self._pos_integrator_limit, self._pos_integrator_limit)
        self._ipy = self._clamp(self._ipy + (err_y * dt), -self._pos_integrator_limit, self._pos_integrator_limit)

        v_sp_x = -(self._kp_pos * err_x + self._ki_pos * self._ipx)
        v_sp_y = -(self._kp_pos * err_y + self._ki_pos * self._ipy)
        v_sp_x = self._clamp(float(v_sp_x), -self._max_v_sp, self._max_v_sp)
        v_sp_y = self._clamp(float(v_sp_y), -self._max_v_sp, self._max_v_sp)

        self._state = PositionLeashState(
            pos_x=float(self._pos_x),
            pos_y=float(self._pos_y),
            err_x=float(err_x),
            err_y=float(err_y),
            v_sp_x=float(v_sp_x),
            v_sp_y=float(v_sp_y),
        )

        vx_err = float(vx - v_sp_x)
        vy_err = float(vy - v_sp_y)
        return self._inner.update(dt_sec=dt, vx_px_s=vx_err, vy_px_s=vy_err, quality=q)

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(min(hi, max(lo, v)))
