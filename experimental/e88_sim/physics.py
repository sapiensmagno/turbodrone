"""Rigid-body and flight-controller model of the E88.

We never get to command motors or rates: the E88 self-levels on board, so a
stick deflection is a request for a *lean angle*. The model is therefore

    stick -> quantized byte -> 33 Hz transmit -> latency -> target lean angle
          -> second-order attitude response -> tilted thrust -> acceleration
          -> drag + trim + turbulence -> velocity -> position

with altitude and yaw handled by their own first-order loops, because the drone
holds height and heading on its own.

World frame: x = right, y = forward, z = up, all in metres. Attitude:
``roll`` is positive with the right side down (accelerates +x), ``pitch`` is
positive nose-down (accelerates +y), ``yaw`` is positive turning left
(counter-clockwise seen from above).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from e88_sim.config import GRAVITY_M_S2, AirframeParams, EnvironmentParams, SimConfig


@dataclass(frozen=True)
class PhysicsState:
    """Ground truth. The real drone cannot measure any of this -- which is the
    entire reason for simulating: every estimate the autopilot produces can be
    scored against the true value."""

    t: float
    x_m: float
    y_m: float
    z_m: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    # True airframe attitude, vibration included -- this is what the camera
    # rides on, so it is what the renderer must use.
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    # Rates of the *smooth* attitude only. Vibration is deliberately excluded:
    # its instantaneous rate is enormous and meaningless at frame rate, and the
    # apparent flow it causes is a frame-to-frame attitude *difference*, which
    # ``DownwardCamera.ground_truth_flow_px`` computes directly.
    roll_rate_rad_s: float
    pitch_rate_rad_s: float
    yaw_rate_rad_s: float
    vibration_roll_rad: float
    vibration_pitch_rad: float
    # Command actually being acted on right now (post-quantization, post-delay).
    applied_roll: float
    applied_pitch: float
    applied_yaw: float
    applied_throttle: float
    landed: bool

    @property
    def pos_xy_m(self) -> Tuple[float, float]:
        return (float(self.x_m), float(self.y_m))

    @property
    def vel_xy_m_s(self) -> Tuple[float, float]:
        return (float(self.vx_m_s), float(self.vy_m_s))

    @property
    def horizontal_speed_m_s(self) -> float:
        return float(math.hypot(float(self.vx_m_s), float(self.vy_m_s)))


@dataclass(frozen=True)
class _Command:
    roll: float
    pitch: float
    yaw: float
    throttle: float
    takeoff: bool = False
    land: bool = False


_NEUTRAL = _Command(roll=0.0, pitch=0.0, yaw=0.0, throttle=50.0)


def axis_to_byte(v: float) -> int:
    """Mirror of ``e88.drone.E88Drone._axis_to_byte``."""
    v = max(-1.0, min(1.0, float(v)))
    return int(round(128 + (127 * v)))


def byte_to_axis(b: int) -> float:
    return (float(int(b)) - 128.0) / 127.0


def throttle_to_byte(v: float) -> int:
    """Mirror of ``e88.drone.E88Drone._throttle_to_byte``."""
    v = max(0.0, min(100.0, float(v)))
    return int(round(255 * (v / 100.0)))


def byte_to_throttle(b: int) -> float:
    return 100.0 * (float(int(b)) / 255.0)


def body_to_world_rotation(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """R_wb, mapping body axes (x right, y forward, z up) into world axes.

    Built as Rz(yaw) @ Ry(roll) @ Rx(-pitch) so that positive roll tips the
    thrust vector toward +x (right) and positive pitch tips it toward +y
    (forward), matching the sign convention documented at module level.
    """
    cr, sr = math.cos(float(roll_rad)), math.sin(float(roll_rad))
    cp, sp = math.cos(-float(pitch_rad)), math.sin(-float(pitch_rad))
    cy, sy = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return rz @ ry @ rx


class QuadPhysics:
    """Fixed-step integrator for the E88 airframe.

    Call :meth:`submit_command` whenever the autopilot sends sticks, and
    :meth:`step` to advance time. The two are deliberately decoupled: commands
    are latched onto the 33 Hz transmit grid and delayed, exactly as they are on
    the real link, so a control loop running at 20 Hz sees the same
    zero-order-hold staircase it would see in flight.
    """

    def __init__(self, cfg: Optional[SimConfig] = None, *, rng: Optional[np.random.Generator] = None) -> None:
        self._cfg = cfg or SimConfig()
        self._af: AirframeParams = self._cfg.airframe
        self._env: EnvironmentParams = self._cfg.environment
        self._rng = rng if rng is not None else np.random.default_rng(int(self._cfg.seed))

        self._t = 0.0

        env = self._env
        self._pos = np.array([float(env.initial_xy_m[0]), float(env.initial_xy_m[1]), float(env.initial_altitude_m)])
        self._vel = np.array([float(env.initial_velocity_m_s[0]), float(env.initial_velocity_m_s[1]), 0.0])
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = math.radians(float(env.initial_yaw_deg))
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._yaw_rate = 0.0

        # Command pipeline: what the autopilot last asked for, what the 33 Hz
        # transmitter has queued, and what the drone is currently acting on.
        self._submitted = _NEUTRAL
        self._queue: Deque[Tuple[float, _Command]] = deque()
        self._applied = _NEUTRAL
        self._next_tx_t = 0.0

        # Turbulence (Ornstein-Uhlenbeck) and barometric bobbing.
        self._gust = np.zeros(2, dtype=np.float64)
        self._alt_bob = 0.0

        # Airframe vibration. Kept separate from the attitude state because it
        # must reach the camera without reaching the thrust vector -- see
        # AirframeParams.vibration_sigma_deg for why.
        self._vib = np.zeros(2, dtype=np.float64)

        # Auto takeoff/land capture, when the one-shot flag commands are used.
        self._auto_target_alt_m: Optional[float] = None
        self._landing = False
        self._landed = bool(self._pos[2] <= 1e-6)

    # ------------------------------------------------------------------ input

    def submit_command(self, *, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, throttle: float = 50.0) -> None:
        """Set the stick state, quantizing exactly as ``E88Drone.send_cmd`` does."""
        if self._af.quantize_commands:
            roll_v = byte_to_axis(axis_to_byte(roll))
            pitch_v = byte_to_axis(axis_to_byte(pitch))
            yaw_v = byte_to_axis(axis_to_byte(yaw))
            throttle_v = byte_to_throttle(throttle_to_byte(throttle))
        else:
            roll_v = max(-1.0, min(1.0, float(roll)))
            pitch_v = max(-1.0, min(1.0, float(pitch)))
            yaw_v = max(-1.0, min(1.0, float(yaw)))
            throttle_v = max(0.0, min(100.0, float(throttle)))

        dz = float(self._af.stick_deadzone_counts) / 127.0
        if dz > 0.0:
            roll_v = 0.0 if abs(roll_v) < dz else roll_v
            pitch_v = 0.0 if abs(pitch_v) < dz else pitch_v
            yaw_v = 0.0 if abs(yaw_v) < dz else yaw_v

        self._submitted = _Command(roll=roll_v, pitch=pitch_v, yaw=yaw_v, throttle=throttle_v)

    def request_takeoff(self) -> None:
        self._auto_target_alt_m = float(self._af.takeoff_altitude_m)
        self._landing = False
        self._landed = False

    def request_land(self) -> None:
        self._auto_target_alt_m = 0.0
        self._landing = True

    # ------------------------------------------------------------------- step

    def step(self, dt_sec: float) -> PhysicsState:
        """Advance by ``dt_sec``, sub-stepping to the configured physics step."""
        remaining = float(dt_sec)
        if remaining <= 0.0:
            return self.state()

        h = float(self._cfg.physics_dt_sec)
        while remaining > 1e-12:
            step = h if remaining > h else remaining
            self._integrate(step)
            remaining -= step
        return self.state()

    def _integrate(self, dt: float) -> None:
        self._t += dt
        self._pump_command_pipeline()

        cmd = self._applied
        af = self._af

        # --- attitude: second-order response toward the commanded lean angle --
        max_tilt = af.max_tilt_rad()
        roll_target = float(cmd.roll) * max_tilt * float(af.cmd_roll_to_right)
        pitch_target = float(cmd.pitch) * max_tilt * float(af.cmd_pitch_to_forward)

        wn = float(af.attitude_omega_n_rad_s)
        zeta = float(af.attitude_zeta)
        rate_limit = math.radians(float(af.max_tilt_rate_deg_s))

        roll_acc = (wn * wn) * (roll_target - self._roll) - (2.0 * zeta * wn * self._roll_rate)
        pitch_acc = (wn * wn) * (pitch_target - self._pitch) - (2.0 * zeta * wn * self._pitch_rate)
        self._roll_rate = float(np.clip(self._roll_rate + roll_acc * dt, -rate_limit, rate_limit))
        self._pitch_rate = float(np.clip(self._pitch_rate + pitch_acc * dt, -rate_limit, rate_limit))
        self._roll += self._roll_rate * dt
        self._pitch += self._pitch_rate * dt

        # --- yaw ---------------------------------------------------------------
        self._yaw_rate = math.radians(
            float(cmd.yaw) * float(af.max_yaw_rate_deg_s) + float(af.yaw_drift_deg_s)
        )
        self._yaw += self._yaw_rate * dt

        # --- horizontal acceleration from the tilted thrust vector -------------
        # While the drone holds altitude the vertical thrust component balances
        # gravity, so the horizontal component is exactly g * (tx/tz, ty/tz).
        r_wb = body_to_world_rotation(self._roll, self._pitch, self._yaw)
        thrust_dir = r_wb @ np.array([0.0, 0.0, 1.0])
        tz = float(max(0.2, thrust_dir[2]))
        accel = np.array(
            [GRAVITY_M_S2 * float(thrust_dir[0]) / tz, GRAVITY_M_S2 * float(thrust_dir[1]) / tz],
            dtype=np.float64,
        )

        # --- drag, trim and turbulence ----------------------------------------
        vel_xy = self._vel[:2]
        speed = float(np.linalg.norm(vel_xy))
        accel -= float(af.drag_linear_1_s) * vel_xy
        accel -= float(af.drag_quadratic_1_m) * speed * vel_xy

        env = self._env
        accel += np.array([float(env.trim_accel_x_m_s2), float(env.trim_accel_y_m_s2)], dtype=np.float64)
        accel += self._step_gust(dt)
        accel += self._pulse_gust()

        if self._landed:
            # On the ground the drone does not slide around; only a takeoff
            # request gets it moving again.
            self._vel[:2] = 0.0
        else:
            self._vel[:2] = vel_xy + accel * dt
            self._pos[:2] = self._pos[:2] + self._vel[:2] * dt

        # --- vertical ----------------------------------------------------------
        self._step_vertical(dt, cmd)
        self._step_vibration(dt)

    def _step_vibration(self, dt: float) -> None:
        sigma = math.radians(float(self._af.vibration_sigma_deg))
        if sigma <= 0.0:
            self._vib[:] = 0.0
            return
        tau = max(1e-4, float(self._af.vibration_tau_sec))
        a = math.exp(-dt / tau)
        self._vib = a * self._vib + sigma * math.sqrt(max(0.0, 1.0 - a * a)) * self._rng.standard_normal(2)

    def _pump_command_pipeline(self) -> None:
        """Latch sticks onto the 33 Hz transmit grid, then apply after latency."""
        interval = max(1e-4, float(self._af.control_interval_sec))
        while self._t >= self._next_tx_t - 1e-12:
            self._queue.append((self._next_tx_t + float(self._af.command_latency_sec), self._submitted))
            self._next_tx_t += interval

        while self._queue and self._queue[0][0] <= self._t + 1e-12:
            _, cmd = self._queue.popleft()
            self._applied = cmd

    def _step_gust(self, dt: float) -> np.ndarray:
        env = self._env
        tau = max(1e-3, float(env.gust_tau_sec))
        sigma = float(env.gust_sigma_m_s2)
        if sigma <= 0.0:
            self._gust[:] = 0.0
            return self._gust
        # Exact discretization of an OU process, so the steady-state std is
        # sigma regardless of the integration step.
        a = math.exp(-dt / tau)
        noise_std = sigma * math.sqrt(max(0.0, 1.0 - a * a))
        self._gust = a * self._gust + noise_std * self._rng.standard_normal(2)
        return self._gust

    def _pulse_gust(self) -> np.ndarray:
        env = self._env
        dur = float(env.gust_pulse_duration_sec)
        if dur <= 0.0:
            return np.zeros(2, dtype=np.float64)
        t0 = float(env.gust_pulse_start_sec)
        if t0 <= self._t < (t0 + dur):
            return np.array(
                [float(env.gust_pulse_accel_m_s2[0]), float(env.gust_pulse_accel_m_s2[1])], dtype=np.float64
            )
        return np.zeros(2, dtype=np.float64)

    def _step_vertical(self, dt: float, cmd: _Command) -> None:
        af = self._af
        z_dyn = float(self._pos[2]) - self._alt_bob

        if self._auto_target_alt_m is not None:
            err = float(self._auto_target_alt_m) - z_dyn
            climb_cmd = float(np.clip(err * float(af.auto_altitude_kp_1_s), -af.max_climb_rate_m_s, af.max_climb_rate_m_s))
            if abs(err) < 0.05 and not self._landing:
                self._auto_target_alt_m = None
        else:
            dev_pct = float(cmd.throttle) - float(af.hover_throttle_pct)
            if abs(dev_pct) < float(af.throttle_deadband_pct):
                dev_pct = 0.0
            climb_cmd = float(np.clip(dev_pct / 50.0, -1.0, 1.0)) * float(af.max_climb_rate_m_s)

        tau = max(1e-3, float(af.climb_tau_sec))
        self._vel[2] += (climb_cmd - float(self._vel[2])) * (dt / tau)
        z_dyn += float(self._vel[2]) * dt

        # Barometric altitude-hold bobbing, as an OU offset on true height.
        bob_tau = max(1e-3, float(af.altitude_hold_tau_sec))
        bob_sigma = float(af.altitude_hold_sigma_m)
        if bob_sigma > 0.0:
            a = math.exp(-dt / bob_tau)
            self._alt_bob = a * self._alt_bob + bob_sigma * math.sqrt(max(0.0, 1.0 - a * a)) * float(
                self._rng.standard_normal()
            )
        else:
            self._alt_bob = 0.0

        if z_dyn <= 0.0:
            z_dyn = 0.0
            self._vel[2] = 0.0
            if self._landing:
                self._landed = True
                self._auto_target_alt_m = None
                self._landing = False
        elif z_dyn > 0.02:
            self._landed = False

        self._pos[2] = max(0.0, z_dyn + self._alt_bob)

    # ------------------------------------------------------------------ output

    def state(self) -> PhysicsState:
        return PhysicsState(
            t=float(self._t),
            x_m=float(self._pos[0]),
            y_m=float(self._pos[1]),
            z_m=float(self._pos[2]),
            vx_m_s=float(self._vel[0]),
            vy_m_s=float(self._vel[1]),
            vz_m_s=float(self._vel[2]),
            roll_rad=float(self._roll) + float(self._vib[0]),
            pitch_rad=float(self._pitch) + float(self._vib[1]),
            yaw_rad=float(self._yaw),
            roll_rate_rad_s=float(self._roll_rate),
            pitch_rate_rad_s=float(self._pitch_rate),
            yaw_rate_rad_s=float(self._yaw_rate),
            vibration_roll_rad=float(self._vib[0]),
            vibration_pitch_rad=float(self._vib[1]),
            applied_roll=float(self._applied.roll),
            applied_pitch=float(self._applied.pitch),
            applied_yaw=float(self._applied.yaw),
            applied_throttle=float(self._applied.throttle),
            landed=bool(self._landed),
        )

    @property
    def t(self) -> float:
        return float(self._t)

    @property
    def cfg(self) -> SimConfig:
        return self._cfg
