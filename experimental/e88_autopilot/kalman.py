from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KalmanEstimate:
    t: float
    x_px: float
    y_px: float
    vx_px_s: float
    vy_px_s: float
    p_vx: float
    p_vy: float


class VelocityKalman2D:
    def __init__(
        self,
        *,
        sigma_a: float = 25.0,
        sigma_v_meas: float = 60.0,
        init_pos_var: float = 1e6,
        init_vel_var: float = 1e4,
        min_quality: float = 0.05,
    ) -> None:
        self._sigma_a = float(sigma_a)
        self._sigma_v_meas = float(sigma_v_meas)
        self._init_pos_var = float(init_pos_var)
        self._init_vel_var = float(init_vel_var)
        self._min_quality = float(min_quality)

        self._t: Optional[float] = None
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._p = np.eye(4, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x[:] = 0.0
        self._p[:] = 0.0
        self._p[0, 0] = self._init_pos_var
        self._p[1, 1] = self._init_pos_var
        self._p[2, 2] = self._init_vel_var
        self._p[3, 3] = self._init_vel_var

    def update_velocity(self, *, t: float, vx_px_s: float, vy_px_s: float, quality: float = 1.0) -> KalmanEstimate:
        tt = float(t)
        if self._t is None:
            self._t = tt
            self._x[2, 0] = float(vx_px_s)
            self._x[3, 0] = float(vy_px_s)
            return self._estimate()

        dt = tt - float(self._t)
        if dt <= 1e-6:
            return self._estimate()

        self._predict(dt)
        self._t = tt

        q = float(max(self._min_quality, min(1.0, quality)))
        r = (self._sigma_v_meas ** 2) / q
        r_mat = np.array([[r, 0.0], [0.0, r]], dtype=np.float64)

        z = np.array([[float(vx_px_s)], [float(vy_px_s)]], dtype=np.float64)
        h = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

        y = z - (h @ self._x)
        s = (h @ self._p @ h.T) + r_mat
        k = self._p @ h.T @ np.linalg.inv(s)

        self._x = self._x + (k @ y)
        i = np.eye(4, dtype=np.float64)
        self._p = (i - (k @ h)) @ self._p

        return self._estimate()

    def _predict(self, dt: float) -> None:
        d = float(dt)
        f = np.array(
            [[1.0, 0.0, d, 0.0], [0.0, 1.0, 0.0, d], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        sa2 = self._sigma_a ** 2
        d2 = d * d
        d3 = d2 * d
        d4 = d2 * d2
        q = sa2 * np.array(
            [[d4 / 4.0, 0.0, d3 / 2.0, 0.0], [0.0, d4 / 4.0, 0.0, d3 / 2.0], [d3 / 2.0, 0.0, d2, 0.0], [0.0, d3 / 2.0, 0.0, d2]],
            dtype=np.float64,
        )

        self._x = f @ self._x
        self._p = (f @ self._p @ f.T) + q

    def _estimate(self) -> KalmanEstimate:
        assert self._t is not None
        return KalmanEstimate(
            t=float(self._t),
            x_px=float(self._x[0, 0]),
            y_px=float(self._x[1, 0]),
            vx_px_s=float(self._x[2, 0]),
            vy_px_s=float(self._x[3, 0]),
            p_vx=float(self._p[2, 2]),
            p_vy=float(self._p[3, 3]),
        )

    def state(self) -> Tuple[float, float, float, float]:
        return float(self._x[0, 0]), float(self._x[1, 0]), float(self._x[2, 0]), float(self._x[3, 0])
