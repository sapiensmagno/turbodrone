"""Deterministic, faster-than-real-time closed-loop simulation.

This mirrors the hold loop of ``AutoStabilizer.run`` step for step, using the
**real** ``VelocityMeasurementConditioner``, ``VelocityKalman2D`` and
``VelocityHoldController`` -- including the axis swap that feeds measured ``vy``
into roll and measured ``vx`` into pitch (``autostabilizer.py:894``). Only the
optical-flow estimator is replaced, by the analytic surrogate in
``flow_sensor.py``.

Why a second implementation of the loop exists at all: the real loop paces
itself with ``time.monotonic()`` and ``time.sleep()`` and reads frames off a
background thread, so it runs at exactly real time and no faster. Sweeping
gains, latencies and altitudes across seeds needs hundreds of runs. This
version drives the same components from a virtual clock, so a 20-second flight
takes about 40 ms.

The duplication is a real risk -- a divergence between this loop and the real
one would produce confident, wrong advice -- so it is pinned down by
``tests/test_sim_fast_loop.py``, which flies the same scenario both ways and
asserts the flight statistics agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from e88_autopilot.autostabilizer import StabilizerConfig
from e88_autopilot.controller import VelocityHoldController
from e88_autopilot.kalman import VelocityKalman2D
from e88_autopilot.measurement_conditioner import VelocityMeasurementConditioner
from e88_sim.camera import DownwardCamera
from e88_sim.config import SimConfig
from e88_sim.flow_sensor import AnalyticFlowSensor
from e88_sim.ground import make_ground
from e88_sim.metrics import HoverMetrics, evaluate
from e88_sim.physics import PhysicsState, QuadPhysics


@dataclass
class FlightLog:
    """Per-control-iteration record, with ground truth alongside every estimate."""

    t: List[float] = field(default_factory=list)
    x_m: List[float] = field(default_factory=list)
    y_m: List[float] = field(default_factory=list)
    z_m: List[float] = field(default_factory=list)
    vx_m_s: List[float] = field(default_factory=list)
    vy_m_s: List[float] = field(default_factory=list)
    roll_rad: List[float] = field(default_factory=list)
    pitch_rad: List[float] = field(default_factory=list)
    cmd_roll: List[float] = field(default_factory=list)
    cmd_pitch: List[float] = field(default_factory=list)
    meas_vx_px_s: List[float] = field(default_factory=list)
    meas_vy_px_s: List[float] = field(default_factory=list)
    truth_vx_px_s: List[float] = field(default_factory=list)
    truth_vy_px_s: List[float] = field(default_factory=list)
    used_vx_px_s: List[float] = field(default_factory=list)
    used_vy_px_s: List[float] = field(default_factory=list)
    frame_age_sec: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def flow_error_px_s(self) -> List[float]:
        return [m - t for m, t in zip(self.meas_vx_px_s, self.truth_vx_px_s)] + [
            m - t for m, t in zip(self.meas_vy_px_s, self.truth_vy_px_s)
        ]

    def metrics(self, *, max_cmd: float, open_loop: Optional[HoverMetrics] = None, settle_sec: float = 2.0) -> HoverMetrics:
        return evaluate(
            t=self.t,
            x_m=self.x_m,
            y_m=self.y_m,
            vx_m_s=self.vx_m_s,
            vy_m_s=self.vy_m_s,
            cmd_roll=self.cmd_roll,
            cmd_pitch=self.cmd_pitch,
            max_cmd=float(max_cmd),
            open_loop=open_loop,
            flow_error_px_s=self.flow_error_px_s() or None,
            settle_sec=float(settle_sec),
        )


@dataclass
class _Frame:
    seq: int
    capture_pose: PhysicsState
    deliver_at: float


def run_fast(
    *,
    sim_cfg: Optional[SimConfig] = None,
    stab_cfg: Optional[StabilizerConfig] = None,
    duration_sec: float = 20.0,
    control: bool = True,
    seed: Optional[int] = None,
    camera: Optional[DownwardCamera] = None,
) -> FlightLog:
    """Fly one scenario on a virtual clock.

    ``control=False`` runs the identical scenario with neutral sticks, which is
    the open-loop baseline every closed-loop result has to be judged against.
    """
    sim = sim_cfg or SimConfig()
    stab = stab_cfg or StabilizerConfig()
    run_seed = int(sim.seed if seed is None else seed)

    rng_phys = np.random.default_rng(run_seed)
    rng_link = np.random.default_rng(run_seed + 1000)
    rng_flow = np.random.default_rng(run_seed + 2000)

    physics = QuadPhysics(sim, rng=rng_phys)
    cam = camera if camera is not None else DownwardCamera(sim.camera, ground=make_ground(sim.ground))
    sensor = AnalyticFlowSensor(
        cam, motion_model=str(stab.flow_motion_model), rng=rng_flow
    )

    conditioner = VelocityMeasurementConditioner(deadband_px_s=float(stab.estimator_deadband_px_s))
    kalman = (
        VelocityKalman2D(sigma_a=float(stab.kalman_sigma_a), sigma_v_meas=float(stab.kalman_sigma_v))
        if bool(stab.use_kalman)
        else None
    )
    controller = VelocityHoldController(
        min_quality=float(stab.min_quality),
        max_cmd=float(stab.max_cmd),
        kp_vx=float(stab.kp_vx),
        kp_vy=float(stab.kp_vy),
        ki_vx=float(stab.ki_vx),
        ki_vy=float(stab.ki_vy),
        deadband_px_s=float(stab.deadband_px_s),
        roll_sign=float(stab.roll_sign),
        pitch_sign=float(stab.pitch_sign),
    )

    log = FlightLog()

    step = max(1e-4, float(sim.physics_dt_sec))
    ctrl_period = 1.0 / max(1.0, float(stab.cmd_rate_hz))
    frame_period = 1.0 / max(1.0, float(sim.video.fps))

    t = 0.0
    next_frame_t = 0.0
    next_ctrl_t = 0.0
    seq = 0

    pending: List[_Frame] = []
    latest: Optional[Tuple[_Frame, float]] = None  # (frame, delivery time)
    last_used_seq = -1
    prev_flow_pose: Optional[PhysicsState] = None
    prev_flow_t: Optional[float] = None
    last_cmd = (0.0, 0.0)

    n_steps = int(round(float(duration_sec) / step))
    for _ in range(n_steps):
        physics.step(step)
        t += step

        # --- camera capture -------------------------------------------------
        if t >= next_frame_t:
            jitter = float(rng_link.normal(0.0, max(0.0, float(sim.video.fps_jitter_sec))))
            next_frame_t = t + max(1e-3, frame_period + jitter)
            if not _frozen(sim, t) and not _dropped(sim, rng_link):
                seq += 1
                latency = max(
                    0.0,
                    float(sim.video.pipeline_latency_sec)
                    + float(rng_link.normal(0.0, max(0.0, float(sim.video.latency_jitter_sec)))),
                )
                pending.append(_Frame(seq=seq, capture_pose=physics.state(), deliver_at=t + latency))

        # --- delivery into the size-1 buffer ---------------------------------
        if pending:
            due = [f for f in pending if f.deliver_at <= t]
            if due:
                pending = [f for f in pending if f.deliver_at > t]
                due.sort(key=lambda f: f.deliver_at)
                latest = (due[-1], due[-1].deliver_at)

        # --- control iteration ------------------------------------------------
        if t < next_ctrl_t:
            continue
        next_ctrl_t = t + ctrl_period

        if not control:
            physics.submit_command(roll=0.0, pitch=0.0, throttle=float(stab.base_throttle))
            _append_open_loop(log, t, physics.state(), stab)
            continue

        if latest is None:
            physics.submit_command(roll=0.0, pitch=0.0, throttle=float(stab.base_throttle))
            last_cmd = (0.0, 0.0)
            continue

        frame, delivered_at = latest
        frame_is_new = frame.seq != last_used_seq
        last_used_seq = frame.seq
        frame_stale_sec = max(0.0, t - delivered_at)

        # autostabilizer.py:695 -- re-running flow on a frame already consumed
        # would fabricate a zero-velocity measurement, so hold instead. Past
        # max_stale_frame_hold_sec the feed is presumed lost and we neutralize.
        if (not frame_is_new) and bool(stab.skip_stale_frames):
            hold_limit = float(stab.max_stale_frame_hold_sec)
            if hold_limit >= 0.0 and frame_stale_sec > hold_limit:
                last_cmd = (0.0, 0.0)
            physics.submit_command(roll=last_cmd[0], pitch=last_cmd[1], throttle=float(stab.base_throttle))
            continue

        if prev_flow_pose is None or prev_flow_t is None:
            prev_flow_pose = frame.capture_pose
            prev_flow_t = t
            physics.submit_command(roll=0.0, pitch=0.0, throttle=float(stab.base_throttle))
            continue

        # autostabilizer.py:757 -- flow dt comes from the loop's own clock, not
        # from the frame timestamp.
        dt_flow = t - prev_flow_t
        est = sensor.measure(prev_flow_pose, frame.capture_pose, dt_flow)
        truth_vx, truth_vy = sensor.truth_px(prev_flow_pose, frame.capture_pose)
        prev_flow_pose = frame.capture_pose
        prev_flow_t = t

        if est is None:
            physics.submit_command(roll=0.0, pitch=0.0, throttle=float(stab.base_throttle))
            last_cmd = (0.0, 0.0)
            continue

        cond = conditioner.apply(vx_px_s=float(est.vx_px_s), vy_px_s=float(est.vy_px_s))
        vx, vy = float(cond.vx_px_s), float(cond.vy_px_s)
        if kalman is not None:
            # autostabilizer.py:810 -- the Kalman is timestamped with the frame
            # timestamp while flow used loop time. Faithfully reproduced.
            k = kalman.update_velocity(t=delivered_at, vx_px_s=vx, vy_px_s=vy, quality=float(est.quality))
            vx, vy = float(k.vx_px_s), float(k.vy_px_s)

        # autostabilizer.py:894 -- the 90 degree camera mount means roll is
        # driven by measured vy and pitch by measured vx.
        out = controller.update(dt_sec=float(est.dt_sec), vx_px_s=vy, vy_px_s=vx, quality=float(est.quality))
        cmd_roll = 0.0 if out is None else float(out.roll)
        cmd_pitch = 0.0 if out is None else float(out.pitch)

        physics.submit_command(roll=cmd_roll, pitch=cmd_pitch, throttle=float(stab.base_throttle))
        last_cmd = (cmd_roll, cmd_pitch)

        s = physics.state()
        log.t.append(t)
        log.x_m.append(float(s.x_m))
        log.y_m.append(float(s.y_m))
        log.z_m.append(float(s.z_m))
        log.vx_m_s.append(float(s.vx_m_s))
        log.vy_m_s.append(float(s.vy_m_s))
        log.roll_rad.append(float(s.roll_rad))
        log.pitch_rad.append(float(s.pitch_rad))
        log.cmd_roll.append(cmd_roll)
        log.cmd_pitch.append(cmd_pitch)
        log.meas_vx_px_s.append(float(est.vx_px_s))
        log.meas_vy_px_s.append(float(est.vy_px_s))
        log.truth_vx_px_s.append(float(truth_vx / max(1e-6, dt_flow)))
        log.truth_vy_px_s.append(float(truth_vy / max(1e-6, dt_flow)))
        log.used_vx_px_s.append(vx)
        log.used_vy_px_s.append(vy)
        log.frame_age_sec.append(frame_stale_sec)

    return log


def _append_open_loop(log: FlightLog, t: float, s: PhysicsState, stab: StabilizerConfig) -> None:
    log.t.append(t)
    log.x_m.append(float(s.x_m))
    log.y_m.append(float(s.y_m))
    log.z_m.append(float(s.z_m))
    log.vx_m_s.append(float(s.vx_m_s))
    log.vy_m_s.append(float(s.vy_m_s))
    log.roll_rad.append(float(s.roll_rad))
    log.pitch_rad.append(float(s.pitch_rad))
    log.cmd_roll.append(0.0)
    log.cmd_pitch.append(0.0)


def _frozen(sim: SimConfig, t: float) -> bool:
    dur = float(sim.video.freeze_duration_sec)
    if dur <= 0.0:
        return False
    start = float(sim.video.freeze_start_sec)
    return bool(start <= t < (start + dur))


def _dropped(sim: SimConfig, rng: np.random.Generator) -> bool:
    p = float(sim.video.drop_probability)
    return bool(p > 0.0 and rng.random() < p)
